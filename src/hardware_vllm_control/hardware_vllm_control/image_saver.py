#!/usr/bin/env python3
import os
import shlex
import subprocess

import rclpy
from rclpy.node import Node

import lgpio

import cv2
import numpy as np


class RpicamTimelapseNode(Node):
    def __init__(self):
        super().__init__('rpicam_timelapse_node')

        # ===== GPIO 설정 =====
        self.led_pins = [23, 27, 21]
        self.gpio_handle = lgpio.gpiochip_open(0)

        for pin in self.led_pins:
            lgpio.gpio_claim_output(self.gpio_handle, pin)
            lgpio.gpio_write(self.gpio_handle, pin, 1)  # 항상 ON

        self.get_logger().info("[GPIO] LED 항상 점등 상태로 설정 완료")

        # ===== 카메라 설정 =====
        self.interval_sec = 7.0
        self.output_dir = os.path.expanduser('~/saved_images')
        self.filename_prefix = 'saved_image_'
        self.filename_ext = '.jpg'

        os.makedirs(self.output_dir, exist_ok=True)

        # 해상도 고정
        self.width = 1920
        self.height = 1080

        # ===== 캘리브레이션 YAML 경로 =====
        self.calib_yaml = os.path.expanduser(
            '/home/ioes/vllm_control_ws/src/hardware_vllm_control/hardware_vllm_control/ship_cam.yaml'
        )

        # ===== undistort 준비 (K, D, map 1회 계산) =====
        self.K, self.D = self.load_calibration(self.calib_yaml)
        self.map1, self.map2, self.newK = self.prepare_undistort_maps(self.K, self.D, self.width, self.height)

        self.get_logger().info(
            f"[CALIB] Loaded: {self.calib_yaml}\n"
            f"  fx={self.K[0,0]:.3f}, fy={self.K[1,1]:.3f}, cx={self.K[0,2]:.3f}, cy={self.K[1,2]:.3f}\n"
            f"[UNDIST] alpha=1.0 (no crop), newK fx={self.newK[0,0]:.3f}, fy={self.newK[1,1]:.3f}"
        )

        self.counter = self.find_start_index()
        self.timer = self.create_timer(self.interval_sec, self.capture_once)

        self.get_logger().info(
            f"[image_saver] 시작: {self.interval_sec}초마다 1장씩 캡처 + undistort 덮어쓰기\n"
            f"저장 경로: {self.output_dir}, 시작 인덱스: {self.counter}"
        )

    # ------------------------
    # Calibration / Undistort
    # ------------------------
    def load_calibration(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Calibration yaml not found: {yaml_path}")

        fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RuntimeError(f"Failed to open calibration yaml: {yaml_path}")

        K = fs.getNode("camera_matrix").mat()
        D = fs.getNode("distortion_coefficients").mat()
        fs.release()

        if K is None or D is None:
            raise RuntimeError("camera_matrix or distortion_coefficients missing in yaml")

        K = np.array(K, dtype=np.float64)
        D = np.array(D, dtype=np.float64).reshape(-1, 1)
        return K, D

    def prepare_undistort_maps(self, K, D, w: int, h: int):
        # alpha=1.0 => 최대한 FOV 유지(검은 테두리 가능), crop 없음
        newK, _roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1.0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newK, (w, h), cv2.CV_16SC2)
        return map1, map2, newK

    # ------------------------
    # File naming / indexing
    # ------------------------
    def find_start_index(self) -> int:
        import re
        pattern = re.compile(rf"{self.filename_prefix}(\d+){self.filename_ext}")
        max_idx = -1

        for f in os.listdir(self.output_dir):
            m = pattern.fullmatch(f)
            if m:
                idx = int(m.group(1))
                max_idx = max(max_idx, idx)

        return max_idx + 1

    def build_output_path(self) -> str:
        filename = f"{self.filename_prefix}{self.counter:06d}{self.filename_ext}"
        return os.path.join(self.output_dir, filename)

    # ------------------------
    # Experiment archive helpers
    # ------------------------
    def get_next_experiment_dir(self) -> str:
        """~/exp_1, ~/exp_2 ... 중 가장 큰 번호를 찾아 +1 폴더 반환"""
        home_dir = os.path.expanduser("~")

        import re
        pattern = re.compile(r"exp_(\d+)")

        max_idx = 0
        for name in os.listdir(home_dir):
            full = os.path.join(home_dir, name)
            if os.path.isdir(full):
                m = pattern.fullmatch(name)
                if m:
                    idx = int(m.group(1))
                    max_idx = max(max_idx, idx)

        next_idx = max_idx + 1
        return os.path.join(home_dir, f"exp_{next_idx}")

    def archive_saved_images(self):
        """노드 종료 시 saved_images 내부 파일을 ~/exp_N으로 이동"""
        try:
            target_dir = self.get_next_experiment_dir()
            os.makedirs(target_dir, exist_ok=True)

            files = os.listdir(self.output_dir)
            if not files:
                self.get_logger().info("[ARCHIVE] saved_images 비어 있음 → 이동 없음")
                return

            self.get_logger().info(f"[ARCHIVE] 보관 폴더 생성/사용: {target_dir}")

            moved = 0
            for f in files:
                src = os.path.join(self.output_dir, f)
                dst = os.path.join(target_dir, f)

                # 파일만 이동 (혹시 디렉토리가 있으면 무시)
                if os.path.isfile(src):
                    os.replace(src, dst)
                    moved += 1

            self.get_logger().info(f"[ARCHIVE] 이동 완료: {moved}개 파일")

        except Exception as e:
            self.get_logger().error(f"[ARCHIVE] 실패: {e}")

    # ------------------------
    # Capture loop
    # ------------------------
    def capture_once(self):
        final_path = self.build_output_path()

        # 임시 파일들(같은 폴더에 두는 게 rename/replace가 안전)
        raw_tmp = final_path.replace(self.filename_ext, f"_raw{self.filename_ext}")
        undist_tmp = final_path.replace(self.filename_ext, f"_undist_tmp{self.filename_ext}")

        cmd = [
            'rpicam-jpeg',
            '-n',
            '-t', '1000',
            '--width', str(self.width),
            '--height', str(self.height),
            '-q', '85',
            '-o', raw_tmp,
        ]

        # self.get_logger().info(f"[image_saver] 캡처 실행: {' '.join(shlex.quote(c) for c in cmd)}")

        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if result.returncode != 0:
                self.get_logger().error(f"[image_saver] 캡처 실패 code={result.returncode}")
                return

            img = cv2.imread(raw_tmp, cv2.IMREAD_COLOR)
            if img is None:
                self.get_logger().error(f"[UNDIST] Failed to read raw image: {raw_tmp}")
                return

            undist = cv2.remap(
                img, self.map1, self.map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

            ok = cv2.imwrite(undist_tmp, undist, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                self.get_logger().error(f"[UNDIST] Failed to write temp undist image: {undist_tmp}")
                return

            # ✅ 최종 파일명으로 "원자적 교체"
            os.replace(undist_tmp, final_path)

            # raw 임시 파일 정리(원본 필요없으니 삭제)
            try:
                os.remove(raw_tmp)
            except Exception:
                pass

            self.get_logger().info(f"[image_saver] 성공(undist overwrite) → {final_path}")
            self.counter += 1

        except Exception as e:
            self.get_logger().error(f"[image_saver] 예외: {e}")

            # 실패 시 남은 tmp 정리 시도
            for p in (raw_tmp, undist_tmp):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    # ------------------------
    # Shutdown / cleanup
    # ------------------------
    def destroy_node(self):
        self.get_logger().info("[NODE] 종료 시작")

        # ✅ 안정성 개선: 타이머 먼저 중지 (종료 중 capture_once 재진입 방지)
        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.cancel()
                self.get_logger().info("[NODE] 타이머 cancel 완료")
        except Exception as e:
            self.get_logger().warn(f"[NODE] 타이머 cancel 실패(무시): {e}")

        # ✅ 종료 시 이미지 보관
        self.get_logger().info("[NODE] saved_images 이미지들 이동")
        self.archive_saved_images()

        # GPIO 정리
        self.get_logger().info("[GPIO] LED OFF 및 GPIO 정리")
        try:
            for pin in self.led_pins:
                lgpio.gpio_write(self.gpio_handle, pin, 0)
            lgpio.gpiochip_close(self.gpio_handle)
        except Exception as e:
            self.get_logger().warn(f"[GPIO] 정리 중 오류(무시): {e}")

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RpicamTimelapseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()