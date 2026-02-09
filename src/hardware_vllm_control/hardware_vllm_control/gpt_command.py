#!/usr/bin/env python3
import os
import re
import base64
import threading
from typing import List, Optional, Tuple, Literal

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from openai import OpenAI
from pydantic import BaseModel


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
            self.get_logger().error(
                "OpenAI API key not found. Please set GPT_API_KEY (or OPENAI_API_KEY)."
            )

        # OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-5.2"

        # Motor command publisher: /cmd_motor (forward/backward/left/right/stop)
        self.move_pub = self.create_publisher(String, "move_direction", 10)
        self.stop_pub = self.create_publisher(String, "stop_robot", 10)

        # Optional busy signal from thrust controller
        self.thrust_busy_sub = self.create_subscription(
            Bool, "thrust_busy", self.thrust_busy_callback, 10
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
        # Prompts (Verbose, paper-ready)
        # =========================

        self.THREAT_PROMPT = (
            "You are an immediate collision risk gate for a small autonomous surface water drone (USV).\n\n"
            "Platform context:\n"
            "- USV size: approx 0.50 m long and 0.265 m wide.\n\n"
            "Camera context:\n"
            "- Camera: Raspberry Pi IMX219 (Camera Module v2 class), wide-angle.\n"
            "- Max still: 3280x2464 (8MP). Typical FoV ~62.2 deg (H) x ~48.8 deg (V).\n"
            "- Wide-angle edge distortion exists; prioritize threats in the central forward corridor.\n"
            "- Image left = USV left.\n\n"
            "Task:\n"
            "Assess ONLY the current image. No memory, no other sensors.\n"
            "Return a single decision: move or stop.\n\n"
            "Rules (be conservative):\n"
            "- If any non-duck obstacle is in the central forward corridor, choose stop.\n"
            "- If an obstacle looks very near (e.g., touches the bottom edge near the center, or occupies a large fraction of the image height), choose stop.\n"
            "- If the yellow duck is centered and looks very near, choose stop.\n"
            "- Otherwise choose move.\n\n"
            "Fallback:\n"
            "If uncertain, choose stop.\n"
        )


        self.DECISION_PROMPT = (
            "You are a navigation decision module for a small autonomous surface water drone (USV).\n\n"
            "Platform context:\n"
            "- USV size: approx 0.50 m long and 0.265 m wide.\n\n"
            "Camera context:\n"
            "- Camera: Raspberry Pi IMX219 (Camera Module v2 class), wide-angle.\n"
            "- Typical FoV ~62.2 deg (H) x ~48.8 deg (V). Edge distortion can stretch objects near borders.\n"
            "- Image left = USV left.\n"
            "- Actions: w=forward, a=turn/steer left, d=turn/steer right, s=reverse.\n\n"
            "Task:\n"
            "From ONLY the current image, output:\n"
            "- decision: move or stop\n"
            "- direction: one of w/a/s/d\n"
            "- duck_found: true/false\n"
            "- duck_position: center or left/right-Ndeg (N in 5,10,15,20,25,30,35,40,45) or unknown\n\n"
            "Duck reporting:\n"
            "- If no duck is clearly visible: duck_found=false and duck_position=unknown.\n"
            "- If visible: duck_found=true and duck_position based on horizontal offset.\n"
            "- Use center for within ±5 degrees.\n"
            "- ASCII only.\n\n"
            "Decision logic:\n"
            "1) Safety first:\n"
            "   - If an obstacle is in the forward corridor or looks very near, set decision=stop and direction=s (or stop).\n"
            "2) If duck is visible:\n"
            "   - Duck left -> move + a, Duck right -> move + d.\n"
            "   - Duck near center and forward corridor clear -> move + w.\n"
            "   - Duck centered and looks very near -> stop.\n"
            "3) If duck is NOT visible:\n"
            "   - Prefer scanning turns (a or d) rather than forward motion, unless the forward corridor is clearly open.\n\n"
            "Fallback:\n"
            "If uncertain, choose stop.\n"
        )


        self.PATH_PROMPT = (
            "You are a short-horizon path planner for a small autonomous surface water drone (USV).\n\n"
            "Platform context:\n"
            "- USV size: approx 0.50 m long and 0.265 m wide.\n\n"
            "Camera context:\n"
            "- Camera: Raspberry Pi IMX219 (Camera Module v2 class), wide-angle.\n"
            "- Typical FoV ~62.2 deg (H) x ~48.8 deg (V). Edge distortion exists.\n"
            "- Image left = USV left.\n"
            "- Actions: w=forward, a=steer/turn left, d=steer/turn right, s=reverse.\n\n"
            "Task:\n"
            "From ONLY the current image, output a short list of actions (w/a/s/d) to approach the yellow duck if visible,\n"
            "while avoiding obstacles.\n\n"
            "Rules:\n"
            "- If duck is not visible: return a short scan path like [\"a\",\"a\"] or [\"d\",\"d\"].\n"
            "- If duck is left: include a actions to bring it toward center.\n"
            "- If duck is right: include d actions to bring it toward center.\n"
            "- After duck is near center and forward corridor is clear: include w to approach.\n"
            "- If obstacles block the forward corridor: steer away first (a/d), or reverse (s) if extremely tight.\n"
            "- If duck is centered and looks very near: return an empty path.\n\n"
            "Fallback:\n"
            "If safe forward motion is not clear, return turn-only actions (a/d).\n"
        )


    def thrust_busy_callback(self, msg: Bool):
        self.thrust_is_busy = msg.data

    def timer_callback(self):
        # 이미 처리 중이거나, 모터 쪽에서 busy라고 알려주면 스킵
        if self.processing or self.thrust_is_busy:
            return
        self.processing = True
        threading.Thread(target=self.main_process, daemon=True).start()


    def publish_motor_command(self, key: str):
        key = key.strip().lower()
        if key in ("w","a","s","d"):
            self.get_logger().info(f"[CMD PUB] move_direction <- {key}")
            self.move_pub.publish(String(data=key))
        elif key == "stop":
            self.get_logger().info("[CMD PUB] stop_robot <- stop")
            self.stop_pub.publish(String(data="stop"))
        else:
            self.get_logger().warn(f"[CMD MAP] Unknown key: {key}")

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
