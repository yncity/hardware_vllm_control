#!/usr/bin/env python3
import os
import re
import base64
import threading
import lgpio
import threading
import time

from typing import List, Optional, Tuple, Literal

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from openai import OpenAI
from pydantic import BaseModel

#GPIO num
GPT_LED_GPIO = 22
GPT_BLINK_DT = 0.5   # 0.5초마다 토글(조금 빠른 반짝)

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
            # 이후 호출이 계속 실패하므로 타이머를 만들지 않도록 중단
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
        # GPT processing LED (GPIO15 blink while calling OpenAI)
        # =========================
        self.gpio_handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.gpio_handle, GPT_LED_GPIO)
        lgpio.gpio_write(self.gpio_handle, GPT_LED_GPIO, 0)  # 기본 OFF

        self._gpt_led_lock = threading.Lock()
        self._gpt_led_refcount = 0
        self._gpt_led_stop_evt = threading.Event()
        self._gpt_led_thread = None
        self._gpt_led_state = 0


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
            "- Images are distortion-corrected.\n"
            "- The camera is mounted on a floating robot, so small motion and vibration may appear in the images due to water movement.\n"
            "- The left side of the image corresponds to the left side of the drone.\n\n"

            "Task:\n"
            "Assess immediate collision risk based only on the current image.\n"
            "Do not assume any information from previous frames or external sensors.\n\n"

            "Rules:\n"
            "- If any obstacle (walls, buoys, floating objects, structures, etc.) appears very close and directly blocks the forward motion, select \"stop\".\n"
            "- If the yellow goal-post target is extremely close and centered, select \"stop\".\n"
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
            "- Camera height above water: about 0.133 m.\n"
            "- Images are distortion-corrected.\n"
            "- The camera is mounted on a floating robot, so small motion and vibration may appear in the images due to water movement.\n"
            "- The left side of the image corresponds to the left side of the drone.\n\n"

            "Target:\n"
            "- The goal is a fixed yellow soccer goal-post.\n"
            "- The goal-post is entirely yellow and has a rectangular frame structure.\n\n"

            "Task:\n"
            "Based only on the current image, decide whether to move or stop, choose a movement direction, "
            "and report goal visibility and approximate position.\n"
            "Do not assume any information from previous frames or external sensors.\n\n"

            "Goal reporting:\n"
            "- If the yellow goal-post is not clearly visible: goal_found=false and goal_position=\"unknown\".\n"
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
            "- Camera height above water: about 0.133 m.\n"
            "- Images are distortion-corrected.\n"
            "- The camera is mounted on a floating robot, so small motion and vibration may appear in the images due to water movement.\n"
            "- The left side of the image corresponds to the left side of the drone.\n\n"

            "Target:\n"
            "- The target is a fixed yellow soccer goal-post.\n\n"

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
    # LED control for GPT processing indication
    # =========================
    
    def _gpt_led_worker(self):
        # stop 이벤트가 set될 때까지 토글
        try:
            while not self._gpt_led_stop_evt.is_set():
                self._gpt_led_state = 0 if self._gpt_led_state else 1
                lgpio.gpio_write(self.gpio_handle, GPT_LED_GPIO, self._gpt_led_state)
                time.sleep(GPT_BLINK_DT)
        finally:
            # 종료 시 OFF로 복귀(원하면 1로 바꿔도 됨)
            self._gpt_led_state = 0
            try:
                lgpio.gpio_write(self.gpio_handle, GPT_LED_GPIO, 0)
            except Exception:
                pass

    def _gpt_led_start(self):
        # 중첩 호출 대응(refcount)
        with self._gpt_led_lock:
            self._gpt_led_refcount += 1
            if self._gpt_led_refcount > 1:
                return

            # 첫 시작일 때만 스레드 시작
            self._gpt_led_stop_evt.clear()
            self._gpt_led_thread = threading.Thread(target=self._gpt_led_worker, daemon=True)
            self._gpt_led_thread.start()

    def _gpt_led_stop(self):
        with self._gpt_led_lock:
            if self._gpt_led_refcount == 0:
                return
            self._gpt_led_refcount -= 1
            if self._gpt_led_refcount > 0:
                return

            # refcount가 0이 될 때만 실제 종료
            self._gpt_led_stop_evt.set()

    try:
        lgpio.gpio_write(self.gpio_handle, GPT_LED_GPIO, 0)
        lgpio.gpiochip_close(self.gpio_handle)
    except Exception:
        pass
        super().destroy_node()

    # GPT LED thread 정리            
    def destroy_node(self):
        try:
            self._gpt_led_stop_evt.set()
            if self._gpt_led_thread is not None:
                self._gpt_led_thread.join(timeout=0.5)
        except Exception:
            pass
    
    def thrust_busy_callback(self, msg: Bool):
        self.thrust_is_busy = msg.data

    def timer_callback(self):
        if self.processing:
            return
        if self.thrust_is_busy:
            self.get_logger().debug("[BUSY] thrust_busy=True -> skip GPT cycle")
            return
        self.processing = True
        threading.Thread(target=self.main_process, daemon=True).start()



    def publish_motor_command(self, key: str):
        key = key.strip().lower()

        # WASD -> cmd_motor 문자열로 변환
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


    def get_latest_image_path(self) -> Optional[str]:
        """
        Find the latest saved_image_<N>.(png|jpg|jpeg) in ~/saved_images
        """
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
        """
        Single-attempt structured call via responses.parse (no custom JSON parsing).
        If the response is incomplete/invalid, this will raise and we return None.
        """
        image_url = self._data_url_from_image(image_path, image_b64)

        # ✅ OpenAI 호출 동안 GPIO15 blink 시작
        self._gpt_led_start()
        
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
                # ✅ A: output token budget up (reduce truncation risk)
                max_output_tokens=max_output_tokens,
                # ✅ D: Pydantic-based parsing/validation
                text_format=text_format_model,
            )

            parsed = resp.output_parsed
            if parsed is None:
                self.get_logger().warn(f"[LLM PARSE][{tag}] output_parsed is None.")
                return None

            # Log parsed object in a stable way
            try:
                self.get_logger().info(f"[LLM PARSED][{tag}] {parsed.model_dump_json()}")
            except Exception:
                self.get_logger().info(f"[LLM PARSED][{tag}] {parsed}")

            return parsed

        except Exception as e:
            self.get_logger().error(f"[LLM ERROR][{tag}] {e}")
            return None

    # -----------------------------
    # Threat check (PATH MODE only)
    # -----------------------------
    def threat_check(self, image_path: str, image_b64: str) -> str:
        self.get_logger().info(f"[THREAT CHECK] Image: {image_path}")

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

    # -----------------------------
    # Decision (NORMAL MODE first)
    # -----------------------------
    def nav_decision(self, image_path: str, image_b64: str) -> Tuple[Optional[str], Optional[str], bool, str]:
        self.get_logger().info(f"[DECISION] Image: {image_path}")

        parsed: Optional[NavOut] = self._call_parsed(
            system_prompt=self.DECISION_PROMPT,
            user_text=(
                "Choose the next navigation action to reach the yellow duck if present, while prioritizing collision avoidance. "
                "If no duck is clearly visible, choose a direction to search."
            ),
            image_path=image_path,
            image_b64=image_b64,
            reasoning_effort="low",
            max_output_tokens=256,  # ✅ A: increased
            text_format_model=NavOut,
            tag="nav_decision",
        )

        if parsed is None:
            return None, None, False, "unknown"

        decision = parsed.decision
        direction = parsed.direction
        duck_found = bool(parsed.duck_found)
        duck_position = str(parsed.duck_position)

        # Enforce logical consistency (post-processing)
        if duck_found is False:
            duck_position = "unknown"
        else:
            if duck_position == "unknown":
                duck_found = False

        return decision, direction, duck_found, duck_position

    # -----------------------------
    # Path planning (short horizon)
    # -----------------------------
    def plan_path_actions(self, image_path: str, image_b64: str) -> List[str]:
        self.get_logger().info(f"[PATH PLAN] Image: {image_path}")

        parsed: Optional[PathOut] = self._call_parsed(
            system_prompt=self.PATH_PROMPT,
            user_text=(
                "Generate a short discrete action sequence to approach the yellow duck if visible, while avoiding obstacles. "
                "Return an empty path if the duck is centered and very close."
            ),
            image_path=image_path,
            image_b64=image_b64,
            reasoning_effort="medium",
            max_output_tokens=256,  # ✅ A: increased
            text_format_model=PathOut,
            tag="path_plan",
        )

        if parsed is None:
            return []

        return [str(a) for a in parsed.path]

    def main_process(self):
        # 레이스 방지: 스레드 시작 직후에도 busy면 즉시 중단
        if self.thrust_is_busy:
            return
        try:
            image_path = self.get_latest_image_path()
            if not image_path or not os.path.exists(image_path):
                self.get_logger().warn("[IMAGE] No image found in ~/saved_images.")
                return

            self.get_logger().info(f"[IMAGE] Using latest image: {image_path}")
            image_b64 = self.image_to_base64(image_path)

            # =====================
            # PATH MODE
            # =====================
            if self.in_path_mode:
                if self.current_step >= len(self.path_actions):
                    self.get_logger().info("[PATH MODE] Completed. Returning to normal mode.")
                    self.in_path_mode = False
                    self.path_actions = []
                    self.current_step = 0
                    return

                # 최신 이미지 다시 반영
                image_path2 = self.get_latest_image_path()
                if not image_path2 or not os.path.exists(image_path2):
                    self.get_logger().warn("[PATH MODE] No image available. Stopping and aborting path mode.")
                    self.in_path_mode = False
                    self.path_actions = []
                    self.current_step = 0
                    self.publish_motor_command("stop")
                    return

                image_b64_2 = self.image_to_base64(image_path2)

                # Threat gate before executing each step (original flow)
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

            # =====================
            # NORMAL MODE (Decision first)
            # =====================
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

            # duck_found == False -> keep searching in normal mode
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
