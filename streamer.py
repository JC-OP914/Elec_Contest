# -*- coding: utf-8 -*-
"""
K230 MJPEG 无线图传模块
=======================
通过 WiFi + HTTP 将检测画面实时推流，浏览器直接观看。
WiFi 连接由系统固件管理，本模块只负责 HTTP Server。
"""

import socket
import network
import time

MJPEG_HEADER = (
    "HTTP/1.1 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
    "Connection: close\r\n"
    "\r\n"
)

BOUNDARY = "\r\n--frame\r\n"
JPEG_FRAME_HEADER = "Content-Type: image/jpeg\r\nContent-Length: {length}\r\n\r\n"


class MJPEGStreamer:
    def __init__(self, port: int = 80):
        self.port = port
        self._server = None
        self._client = None
        self._header_sent = False

    def get_ip(self):
        """获取当前 WiFi IP (假设系统已连接)"""
        try:
            wlan = network.WLAN(network.STA_IF)
            if wlan.isconnected():
                return wlan.ifconfig()[0]
        except Exception:
            pass
        return None

    def start_server(self):
        """启动 TCP 服务器"""
        try:
            addr = socket.getaddrinfo("0.0.0.0", self.port)[0][-1]
            self._server = socket.socket()
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind(addr)
            self._server.listen(1)
            self._server.settimeout(0.3)
            ip = self.get_ip()
            print("[streamer] HTTP server on %s:%d" % (ip if ip else "?.?.?.?", self.port))
            return True
        except Exception as e:
            print("[streamer] Server start failed:", e)
            return False

    def accept_client(self):
        """非阻塞接受客户端，读取 HTTP 请求"""
        if self._client:
            return True
        try:
            cl, addr = self._server.accept()
            cl.settimeout(2.0)
            # 读取浏览器发来的 HTTP 请求，否则部分客户端会超时
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

    def send_frame(self, jpeg_data: bytes):
        if not self._client:
            return False
        try:
            if not self._header_sent:
                self._client.send(MJPEG_HEADER)
                self._header_sent = True
            self._client.send(BOUNDARY)
            self._client.send(JPEG_FRAME_HEADER.format(length=len(jpeg_data)))
            self._client.send(jpeg_data)
            return True
        except Exception as e:
            print("[streamer] Client lost:", e)
            self._close_client()
            return False

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
        print("[streamer] Closed")
