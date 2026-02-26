#!/usr/bin/env python3
import os
import re
import base64
import threading
import time
from typing import List, Optional, Tuple, Literal

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from openai import OpenAI
from pydantic import BaseModel

# =========================
# GPT COMMAND LED (BCM GPIO22)
# - GPT 호출 중: 0.5초 주기로 깜빡임
# - 평소: OFF
# =========================
USE_GPT_LED = True
GPT_LED_GPIO = 22
GPT_BLINK_DT = 0.5

try:
    import lgpio
except Exception:
    lgpio = None
    USE_GPT_LED = False


# =========================
# Pydantic output contracts (Structured Outputs via responses.parse)
# =========================
DuckPos = Literal[
    "unknown", "center",
    "left-5deg", "left-10deg", "left-15deg", "left-20deg", "left-25deg",
    "left-30deg", "left-35deg", "left-40deg", "left-45deg",
    "right-5deg", "right-10deg", "right-15deg", "right-20deg", "right-25deg",
    "right-30deg", "right-35deg", "right-40deg", "right-45deg",
]
MoveStop = Literal["move", "stop"]
WASD = Literal["w", "a", "s", "d"]


class ThreatOut(BaseModel):
    decision: MoveStop


class NavOut(BaseModel):
    decision: MoveStop
    direction: WASD
    duck_found: bool
    duck_position: DuckPos


class PathOut(BaseModel):
    path: List[WASD]


class GPTImageRobotController(Node):
    def __init__(self):
        super().__init__("gpt_image_robot_controller")

        # OpenAI API key
        api_key = os.getenv("GPT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.get_logger().error("OpenAI API key not found. Please set GPT_API_KEY (or OPENAI_API_KEY).")
            raise RuntimeError("Missing OpenAI API key")

        # OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-5.2"

        # Motor command publisher: /cmd_motor (forward/backward/left/right/stop)
        self.cmd_pub = self.create_publisher(String, "/cmd_motor", 10)

        # Optional busy signal from thrust controller
        self.thrust_busy_sub = self.create_subscription(
            Bool, "/thrust_busy", self.thrust_busy_callback, 10
        )
        
        # 런타임 측정 시작 시각(ROS time)
        self._run_start = self.get_clock().now()
        self.get_logger().info(f"[RUNTIME] start_time={self._run_start.nanoseconds} (ns)")

        # Path mode state
        self.in_path_mode = False
        self.path_actions: List[str] = []
        self.current_step = 0

        # State flags
        self.thrust_is_busy = False
        self.processing = False

        # Loop timer (seconds)
        self.timer_period = 5.0
        self.timer = self.create_timer(self.timer_period, self.timer_callback)


        # =========================
        # Prompts (Verbose, paper-ready)
        # =========================

        self.THREAT_PROMPT = (
            "You are an immediate collision risk assessment module for a small autonomous surface water drone.\n\n"
            "Platform and camera context:\n"
            "- The drone is a small real-world twin-hull surface vehicle (catamaran).\n"
            "- Approximate size: 0.50 m long and 0.265 m wide.\n"
            "- Camera: Raspberry Pi IMX219.\n"
            "- Image resolution: 1920x1080.\n"
            "- Camera height above water: about 0.133 m.\n"
            "- Camera intrinsics (calibrated): fx=1483.34, fy=744.90.\n"
            "- Field of view (from intrinsics): HFOV~65.8 deg, VFOV~71.9 deg (Diag~88.4 deg).\n"
            "- Images are distortion-corrected.\n"
            "- The camera is mounted on a floating robot, so small motion and vibration may appear in the images due to water movement.\n"
            "- The red structures and LED lights visible at the bottom of the image belong to the robot itself.\n"
            "- They are part of the robot body and should never be considered as obstacles.\n"
            "- These parts always appear in the camera view and do not affect navigation or collision risk.\n"
            "- The left side of the image corresponds to the left side of the drone.\n\n"
            "Task:\n"
            "Assess immediate collision risk based only on the current image.\n"
            "Do not assume any information from previous frames or external sensors.\n\n"
            "Rules:\n"
            "- If any obstacle (walls, buoys, floating objects, structures, etc.) appears very close and directly blocks the forward motion, select \"stop\".\n"
            "- If the yellow target is extremely close and centered, select \"stop\".\n"
            "- Otherwise, select \"move\".\n\n"
            "Important:\n"
            "- Use visual reasoning to estimate proximity.\n"
            "- Do not rely on exact physical distance or pixel thresholds.\n"
            "- Be conservative but avoid unnecessary stopping.\n\n"
            "Fallback:\n"
            "If uncertain, select \"move\".\n"
        )

        self.DECISION_PROMPT = (
            "You are a navigation decision module for a small real-world autonomous surface water drone.\n\n"
            "Platform and camera context:\n"
            "- The drone is a small twin-hull (catamaran-style) robot.\n"
            "- Approximate size: 0.50 m long and 0.265 m wide.\n"
            "- Camera: Raspberry Pi IMX219.\n"
            "- Image resolution: 1920x1080.\n"
            "- Camera intrinsics (calibrated): fx=1483.34, fy=744.90.\n"
            "- Field of view (from intrinsics): HFOV~65.8 deg, VFOV~71.9 deg (Diag~88.4 deg).\n"
            "- Camera height above water: about 0.133 m.\n"
            "- Images are distortion-corrected.\n"
            "- The camera is mounted on a floating robot, so small motion and vibration may appear in the images due to water movement.\n"
            "- The red structures and LED lights visible at the bottom of the image belong to the robot itself.\n"
            "- They are part of the robot body and should never be considered as obstacles.\n"
            "- These parts always appear in the camera view and do not affect navigation or collision risk.\n"
            "- The left side of the image corresponds to the left side of the drone.\n\n"
            "Target:\n"
            "- The goal is a yellow buoy.\n"
            "- The buoy is entirely yellow and has a round frame structure.\n\n"
            "Task:\n"
            "Based only on the current image, decide whether to move or stop, choose a movement direction, "
            "and report goal visibility and approximate position.\n"
            "Do not assume any information from previous frames or external sensors.\n\n"
            "Goal reporting:\n"
            "- If the yellow buoy is not clearly visible: goal_found=false and goal_position=\"unknown\".\n"
            "- If visible: goal_found=true and goal_position must be:\n"
            "  \"center\" OR \"left-Ndeg\" or \"right-Ndeg\".\n"
            "- Use ASCII only.\n\n"
            "Decision rules:\n"
            "1. Safety has the highest priority.\n"
            "2. If there are no visible obstacles and the goal is not visible, rotate or explore to search.\n"
            "3. If the goal is visible and far, approach it while avoiding obstacles.\n"
            "4. If the goal is centered and very close, select \"stop\".\n"
            "5. If obstacles appear close, adjust direction to avoid them before approaching the goal.\n\n"
            "Important:\n"
            "- Use visual reasoning.\n"
            "- Do not use fixed thresholds.\n"
            "- Prefer smooth and safe navigation.\n\n"
            "Fallback:\n"
            "If uncertain, choose a conservative direction.\n"
        )

        self.PATH_PROMPT = (
            "You are a short-horizon path planning module for a small real-world autonomous surface water drone.\n\n"
            "Platform and camera context:\n"
            "- Twin-hull robot, about 0.50 m long and 0.265 m wide.\n"
            "- Camera: Raspberry Pi IMX219.\n"
            "- Image resolution: 1920x1080.\n"
            "- Camera intrinsics (calibrated): fx=1483.34, fy=744.90.\n"
            "- Field of view (from intrinsics): HFOV~65.8 deg, VFOV~71.9 deg (Diag~88.4 deg).\n"
            "- Camera height above water: about 0.133 m.\n"
            "- Images are distortion-corrected.\n"
            "- The camera is mounted on a floating robot, so small motion and vibration may appear in the images due to water movement.\n"
            "- The red structures and LED lights visible at the bottom of the image belong to the robot itself.\n"
            "- They are part of the robot body and should never be considered as obstacles.\n"
            "- These parts always appear in the camera view and do not affect navigation or collision risk.\n"
            "- The left side of the image corresponds to the left side of the drone.\n\n"
            "Target:\n"
            "- The target is a yellow buoy.\n\n"
            "Task:\n"
            "Based only on the current image, output a short sequence of actions to approach the goal safely.\n\n"
            "Rules:\n"
            "- If the goal is visible on the left, include \"a\" actions.\n"
            "- If the goal is visible on the right, include \"d\" actions.\n"
            "- When the goal is near the center and the path looks safe, include \"w\".\n"
            "- If obstacles appear close, avoid them first.\n"
            "- If the goal is extremely close, return an empty path.\n\n"
            "Priority:\n"
            "Collision avoidance has higher priority.\n\n"
            "Fallback:\n"
            "Rotate to search for a safer direction.\n"
        )


        # =========================
        # GPT LED (GPIO22)
        # =========================
        self._led_enabled = False
        self._led_handle = None
        self._led_state = 0
        self._led_lock = threading.Lock()
        self._led_refcount = 0
        self._led_stop_evt = threading.Event()
        self._led_thread = None

        if USE_GPT_LED and (lgpio is not None):
            self._init_gpt_led()
        else:
            self.get_logger().info("[GPT LED] disabled (lgpio unavailable)")

    # =========================
    # GPIO22 LED: init / worker / start-stop / cleanup
    # =========================
    def _init_gpt_led(self):
        try:
            self._led_handle = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._led_handle, GPT_LED_GPIO)
            lgpio.gpio_write(self._led_handle, GPT_LED_GPIO, 1)  # default OFF
            self._led_enabled = True
            self.get_logger().info(f"[GPT LED] enabled on GPIO{GPT_LED_GPIO}")
        except Exception as e:
            self._led_enabled = False
            try:
                if self._led_handle is not None:
                    lgpio.gpiochip_close(self._led_handle)
            except Exception:
                pass
            self._led_handle = None
            self.get_logger().warn(f"[GPT LED] disabled (GPIO busy/unavailable): {e}")

    def _led_worker(self):
        try:
            while not self._led_stop_evt.is_set():
                self._led_state = 0 if self._led_state else 1
                try:
                    lgpio.gpio_write(self._led_handle, GPT_LED_GPIO, self._led_state)
                except Exception:
                    pass
                time.sleep(GPT_BLINK_DT)
        finally:
            # turn OFF on exit
            self._led_state = 0
            try:
                lgpio.gpio_write(self._led_handle, GPT_LED_GPIO, 1)
            except Exception:
                pass

    def _led_start(self):
        if not self._led_enabled:
            return
        with self._led_lock:
            self._led_refcount += 1
            if self._led_refcount > 1:
                return
            self._led_stop_evt.clear()
            self._led_thread = threading.Thread(target=self._led_worker, daemon=True)
            self._led_thread.start()

    def _led_stop(self):
        if not self._led_enabled:
            return
        with self._led_lock:
            if self._led_refcount == 0:
                return
            self._led_refcount -= 1
            if self._led_refcount > 0:
                return
            self._led_stop_evt.set()

    def destroy_node(self):
        # 1) 런타임 측정 종료 시각(ROS time) - 가장 먼저!
        try:
            end = self.get_clock().now()
            dt_ns = end.nanoseconds - self._run_start.nanoseconds
            dt_s = dt_ns / 1e9
            self.get_logger().info(
                f"[RUNTIME] end_time={end.nanoseconds} (ns), elapsed={dt_s:.3f}s"
            )
        except Exception as e:
            try:
                self.get_logger().warn(f"[RUNTIME] failed to compute elapsed time: {e}")
            except Exception:
                pass

        # 2) stop LED thread + close gpio
        try:
            self._led_stop_evt.set()
        except Exception:
            pass

        if self._led_enabled and (self._led_handle is not None):
            try:
                lgpio.gpio_write(self._led_handle, GPT_LED_GPIO, 1)  # OFF로 맞추기 (init과 동일)
            except Exception:
                pass
            try:
                lgpio.gpiochip_close(self._led_handle)
            except Exception:
                pass
            self._led_handle = None
            self._led_enabled = False

        return super().destroy_node()

    # =========================
    # ROS callbacks
    # =========================
    def thrust_busy_callback(self, msg: Bool):
        self.thrust_is_busy = bool(msg.data)

    def timer_callback(self):
        if self.processing:
            return
        if self.thrust_is_busy:
            self.get_logger().debug("[BUSY] thrust_busy=True -> skip GPT cycle")
            return
        self.processing = True
        threading.Thread(target=self.main_process, daemon=True).start()

    # =========================
    # Motor publishing
    # =========================
    def publish_motor_command(self, key: str):
        key = key.strip().lower()
        key_to_cmd = {
            "w": "forward",
            "a": "left",
            "s": "backward",
            "d": "right",
            "stop": "stop",
        }
        cmd = key_to_cmd.get(key)
        if cmd is None:
            self.get_logger().warn(f"[CMD MAP] Unknown key: {key}")
            return
        self.get_logger().info(f"[CMD PUB] /cmd_motor <- {cmd}")
        self.cmd_pub.publish(String(data=cmd))

    # =========================
    # Image helpers
    # =========================
    def get_latest_image_path(self) -> Optional[str]:
        image_dir = os.path.expanduser("~/saved_images")
        pattern = re.compile(r"saved_image_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
        try:
            candidates = []
            for f in os.listdir(image_dir):
                m = pattern.fullmatch(f)
                if m:
                    candidates.append((int(m.group(1)), f))
            if not candidates:
                return None
            latest = max(candidates)[1]
            return os.path.join(image_dir, latest)
        except Exception as e:
            self.get_logger().warn(f"[IMAGE] Failed to get latest image: {e}")
            return None

    def image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _data_url_from_image(self, image_path: str, image_b64: str) -> str:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        elif ext in (".jpg", ".jpeg"):
            mime = "image/jpeg"
        else:
            mime = "image/jpeg"
        return f"data:{mime};base64,{image_b64}"

    # =========================
    # OpenAI call (with GPIO22 blink during call)
    # =========================
    def _call_parsed(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_path: str,
        image_b64: str,
        reasoning_effort: str,
        max_output_tokens: int,
        text_format_model,
        tag: str,
    ):
        image_url = self._data_url_from_image(image_path, image_b64)

        # GPT 호출 시작 -> LED blink start
        self._led_start()
        try:
            resp = self.client.responses.parse(
                model=self.model_name,
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": image_url},
                            {"type": "input_text", "text": user_text},
                        ],
                    },
                ],
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                text_format=text_format_model,
            )

            parsed = resp.output_parsed
            if parsed is None:
                self.get_logger().warn(f"[LLM PARSE][{tag}] output_parsed is None.")
                return None

            try:
                self.get_logger().info(f"[LLM PARSED][{tag}] {parsed.model_dump_json()}")
            except Exception:
                self.get_logger().info(f"[LLM PARSED][{tag}] {parsed}")

            return parsed

        except Exception as e:
            self.get_logger().error(f"[LLM ERROR][{tag}] {e}")
            return None

        finally:
            # GPT 호출 종료 -> LED blink stop
            self._led_stop()

    # =========================
    # Decision modules
    # =========================
    def threat_check(self, image_path: str, image_b64: str) -> str:
        parsed: Optional[ThreatOut] = self._call_parsed(
            system_prompt=self.THREAT_PROMPT,
            user_text="Assess immediate collision risk based only on this image and return a single decision.",
            image_path=image_path,
            image_b64=image_b64,
            reasoning_effort="low",
            max_output_tokens=120,
            text_format_model=ThreatOut,
            tag="threat_check",
        )
        if parsed is None:
            self.get_logger().warn("[THREAT CHECK] No valid parsed output. Defaulting to STOP.")
            return "stop"
        return parsed.decision

    def nav_decision(self, image_path: str, image_b64: str) -> Tuple[Optional[str], Optional[str], bool, str]:
        parsed: Optional[NavOut] = self._call_parsed(
            system_prompt=self.DECISION_PROMPT,
            user_text="Decide move/stop and direction (w/a/s/d). Report duck_found and duck_position.",
            image_path=image_path,
            image_b64=image_b64,
            reasoning_effort="low",
            max_output_tokens=256,
            text_format_model=NavOut,
            tag="nav_decision",
        )
        if parsed is None:
            return None, None, False, "unknown"

        decision = parsed.decision
        direction = parsed.direction
        duck_found = bool(parsed.duck_found)
        duck_position = str(parsed.duck_position)

        if duck_found is False:
            duck_position = "unknown"
        else:
            if duck_position == "unknown":
                duck_found = False

        return decision, direction, duck_found, duck_position

    def plan_path_actions(self, image_path: str, image_b64: str) -> List[str]:
        parsed: Optional[PathOut] = self._call_parsed(
            system_prompt=self.PATH_PROMPT,
            user_text="Generate short action sequence (w/a/s/d). Empty path if already at goal.",
            image_path=image_path,
            image_b64=image_b64,
            reasoning_effort="medium",
            max_output_tokens=256,
            text_format_model=PathOut,
            tag="path_plan",
        )
        if parsed is None:
            return []
        return [str(a) for a in parsed.path]

    # =========================
    # Main loop logic
    # =========================
    def main_process(self):
        if self.thrust_is_busy:
            self.processing = False
            return

        try:
            image_path = self.get_latest_image_path()
            if not image_path or not os.path.exists(image_path):
                self.get_logger().warn("[IMAGE] No image found in ~/saved_images.")
                return

            image_b64 = self.image_to_base64(image_path)

            # PATH MODE
            if self.in_path_mode:
                if self.current_step >= len(self.path_actions):
                    self.get_logger().info("[PATH MODE] Completed. Returning to normal mode.")
                    self.in_path_mode = False
                    self.path_actions = []
                    self.current_step = 0
                    return

                image_path2 = self.get_latest_image_path()
                if not image_path2 or not os.path.exists(image_path2):
                    self.get_logger().warn("[PATH MODE] No image available. Aborting path mode.")
                    self.in_path_mode = False
                    self.path_actions = []
                    self.current_step = 0
                    self.publish_motor_command("stop")
                    return

                image_b64_2 = self.image_to_base64(image_path2)

                gate = self.threat_check(image_path2, image_b64_2)
                if gate == "stop":
                    self.get_logger().warn("[PATH MODE] Threat gate STOP. Aborting path mode.")
                    self.in_path_mode = False
                    self.path_actions = []
                    self.current_step = 0
                    self.publish_motor_command("stop")
                    return

                cmd = self.path_actions[self.current_step]
                self.get_logger().info(f"[PATH MODE] Step {self.current_step + 1}/{len(self.path_actions)}: {cmd}")
                self.publish_motor_command(cmd)
                self.current_step += 1
                return

            # NORMAL MODE
            decision, direction, duck_found, duck_position = self.nav_decision(image_path, image_b64)
            self.get_logger().info(
                f"[STATE] decision={decision}, direction={direction}, duck_found={duck_found}, duck_position={duck_position}"
            )

            if duck_found:
                self.get_logger().info("[NORMAL MODE] Duck found. Switching to PATH MODE.")
                image_path2 = self.get_latest_image_path()
                if not image_path2 or not os.path.exists(image_path2):
                    self.get_logger().warn("[PATH MODE] No image available at path start. Aborting.")
                    return

                image_b64_2 = self.image_to_base64(image_path2)
                self.path_actions = self.plan_path_actions(image_path2, image_b64_2)
                self.current_step = 0

                if not self.path_actions:
                    self.get_logger().info("[PATH MODE] Empty path. Stopping and returning to normal mode.")
                    self.publish_motor_command("stop")
                    self.in_path_mode = False
                    return

                self.in_path_mode = True
                first = self.path_actions[0]
                self.get_logger().info(f"[PATH MODE] Immediate step 1/{len(self.path_actions)}: {first}")
                self.publish_motor_command(first)
                self.current_step = 1
                return

            # searching in normal mode
            if decision == "stop":
                self.publish_motor_command("stop")
            elif decision == "move" and direction in ("w", "a", "s", "d"):
                self.publish_motor_command(direction)
            else:
                self.get_logger().warn("[NORMAL MODE] Invalid decision output. Defaulting to STOP.")
                self.publish_motor_command("stop")

        except Exception as e:
            self.get_logger().error(f"[ERROR] Unhandled exception: {e}")
        finally:
            self.processing = False


def main(args=None):
    rclpy.init(args=args)
    node = GPTImageRobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()