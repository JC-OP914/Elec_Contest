# -*- coding: utf-8 -*-
"""
K230 MJPEG 无线图传模块
=======================
封装 WiFi 连接 + sensor 通道管理 + MJPEG HTTP 推流。
main.py 只需调用 setup / get_jpeg / send_frame。
"""

import socket
import network
import time
# CAM_CHN_ID_0 等常量在使用时才 import

_MJPEG_HEADER = (
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Connection: close\r\n"
    "\r\n"
)
_BOUNDARY = "\r\n--frame\r\n"
_JPEG_HDR = "Content-Type: image/jpeg\r\nContent-Length: {length}\r\n\r\n"


class MJPEGStreamer:
    def __init__(self, sensor, ssid, password, port=80):
        self._sensor = sensor
        self.ssid = ssid
        self.password = password
        self.port = port
        self._wlan = None
        self._server = None
        self._client = None
        self._header_sent = False

    # ---- Sensor 通道 ----

    def setup_sensor_channel(self):
        """复用已有的 CAM_CHN_ID_0 (320x240)，不需要重新配置 sensor"""
        print("[streamer] Using CAM_CHN_ID_0 for streaming")

    def get_jpeg(self, quality=70):
        """从 CAM_CHN_ID_0 取一帧，返回 JPEG bytes"""
        from media.sensor import CAM_CHN_ID_0
        frame = self._sensor.snapshot(chn=CAM_CHN_ID_0)
        if frame:
            jpeg = frame.compress(quality=quality)
            return bytes(jpeg)
        return None

    # ---- WiFi ----

    def connect_wifi(self):
        self._wlan = network.WLAN(network.STA_IF)
        self._wlan.active(True)
        if self._wlan.isconnected():
            ip = self._wlan.ifconfig()[0]
            print("[streamer] WiFi already connected, IP:", ip)
            return ip
        self._wlan.disconnect()
        time.sleep(0.5)
        self._wlan.connect(self.ssid, self.password)
        for _ in range(25):
            if self._wlan.isconnected():
                break
            time.sleep(1)
            print(".", end="")
        print()
        if self._wlan.isconnected():
            ip = self._wlan.ifconfig()[0]
            print("[streamer] WiFi OK, IP:", ip)
            return ip
        print("[streamer] WiFi FAILED, status:", self._wlan.status())
        return None

    def connected(self):
        return self._wlan and self._wlan.isconnected()

    # ---- HTTP Server ----

    def start_server(self):
        try:
            addr = socket.getaddrinfo("0.0.0.0", self.port)[0][-1]
            self._server = socket.socket()
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind(addr)
            self._server.listen(1)
            self._server.settimeout(0.3)
            ip = self._wlan.ifconfig()[0] if self.connected() else "?.?.?.?"
            print("[streamer] HTTP on %s:%d" % (ip, self.port))
            return True
        except Exception as e:
            print("[streamer] Server failed:", e)
            return False

    def accept_client(self):
        if self._client:
            return True
        try:
            cl, addr = self._server.accept()
            cl.settimeout(2.0)
            try:
                cl.recv(1024)
            except Exception:
                pass
            self._client = cl
            self._header_sent = False
            print("[streamer] Client:", addr)
            return True
        except OSError:
            return False

    # ---- 发送 ----

    def _send_all(self, data):
        if isinstance(data, str):
            data = data.encode()
        buf = data
        while buf:
            try:
                n = self._client.send(buf)
                if n > 0:
                    buf = buf[n:]
                else:
                    time.sleep(0.01)
            except OSError as e:
                if e.args[0] == 11:
                    time.sleep(0.01)
                    continue
                raise

    def send_frame(self, jpeg_data):
        if not self._client:
            return
        try:
            if not self._header_sent:
                self._send_all(_MJPEG_HEADER)
                self._header_sent = True
            self._send_all(_BOUNDARY)
            self._send_all(_JPEG_HDR.format(length=len(jpeg_data)))
            self._send_all(jpeg_data)
        except Exception as e:
            print("[streamer] Client lost:", e)
            self._close_client()

    def _close_client(self):
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self._client = None
        self._header_sent = False

    def close(self):
        self._close_client()
        try:
            if self._server:
                self._server.close()
        except Exception:
            pass
        try:
            if self._wlan:
                self._wlan.disconnect()
                self._wlan.active(False)
        except Exception:
            pass
        print("[streamer] Closed")
