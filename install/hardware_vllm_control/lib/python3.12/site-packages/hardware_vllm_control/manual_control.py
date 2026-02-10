#!/usr/bin/env python3
import sys
import termios
import tty
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


CMD_TOPIC = '/cmd_motor'


class KeyboardCmdNode(Node):
    def __init__(self):
        super().__init__('keyboard_cmd_node')

        self.pub = self.create_publisher(String, CMD_TOPIC, 10)

        self.get_logger().info(
            "키보드 제어 노드 시작.\n"
            "w: forward  | s: backward\n"
            "a: left     | d: right\n"
            "x: stop     | q: 노드 종료\n"
        )

        # 종료 플래그
        self._running = True

        # 키보드 입력을 받는 쓰레드 시작
        self._thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self._thread.start()

    def send_cmd(self, cmd: str):
        msg = String()
        msg.data = cmd
        self.pub.publish(msg)
        self.get_logger().info(f"키 입력 → publish: '{cmd}'")

    # ───────────────────────────────
    # 터미널에서 1글자씩 받는 함수들
    # ───────────────────────────────
    def get_key(self):
        """
        터미널을 raw 모드로 바꿔서 1글자를 블로킹 입력.
        Ctrl+C는 그대로 예외로 나가게 둔다.
        """
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def keyboard_loop(self):
        """
        별도 쓰레드에서 계속 키 입력을 받고, 명령 매핑해서 publish.
        """
        while rclpy.ok() and self._running:
            key = self.get_key()
            key = key.lower()

            if key == 'w':
                self.send_cmd('forward')
            elif key == 's':
                self.send_cmd('backward')
            elif key == 'a':
                self.send_cmd('left')
            elif key == 'd':
                self.send_cmd('right')
            elif key == 'x':
                self.send_cmd('stop')
            elif key == 'q':
                # q 누르면 노드 종료
                self.get_logger().info("q 입력: 노드 종료 요청")
                self._running = False
                # rclpy.shutdown() 은 main 쪽에서 처리
            else:
                # 다른 키는 무시 (엔터 등)
                pass

    def cleanup(self):
        self._running = False
        self.get_logger().info("KeyboardCmdNode 종료.")


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardCmdNode()

    try:
        # spin 하면서 콜백/로그 처리, 키 입력은 별도 쓰레드에서 처리
        while rclpy.ok() and node._running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()