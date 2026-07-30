# -*- coding: utf-8 -*-
"""
UART 通讯模块
=============
二进制帧协议，将 PnP 测距结果 (x 偏移) 发送给电控。

帧格式 (7 bytes):
  字节   | 字段     | 说明
  ------|---------|------------------
  0     | Header0 | 0xAA 帧头
  1     | Header1 | 0x55
  2     | Length  | 0x02 (数据长度)
  3-4   | x_cm    | int16 LE, x_cm * 100
  5     | Checksum| XOR(byte2~byte4)
  6     | Tail    | 0xED 帧尾

示例: x=12.34
  AA 55 02 D2 04 56 ED
          ↑ x=1234=0x04D2  checksum=02^D2^04=D4... wait
  let me recalculate: 0x02 ^ 0xD2 ^ 0x04 = 0xD4
  AA 55 02 D2 04 D4 ED
"""

import time
from machine import UART, FPIOA


class UartComm:
    """UART 二进制帧通讯 (K230 CanMV FPIOA 引脚映射)"""

    def __init__(self, uart_id=3, baudrate=115200,
                 tx_pin=50, rx_pin=51, send_interval_ms=50):
        """
        :param uart_id: UART 编号 (2 或 3)
        :param baudrate: 波特率
        :param tx_pin: TX 引脚 IO 号
        :param rx_pin: RX 引脚 IO 号
        :param send_interval_ms: 最小发送间隔 (ms), 防拥堵
        """
        self.send_interval_ms = send_interval_ms
        self._last_send = 0

        # ---- FPIOA 引脚映射 ----
        fpioa = FPIOA()
        tx_func = getattr(FPIOA, "UART%d_TXD" % uart_id)
        rx_func = getattr(FPIOA, "UART%d_RXD" % uart_id)
        fpioa.set_function(tx_pin, tx_func, oe=1)
        fpioa.set_function(rx_pin, rx_func, ie=1)

        # ---- UART 初始化 ----
        uart_enum = getattr(UART, "UART%d" % uart_id)
        self.uart = UART(
            uart_enum,
            baudrate=baudrate,
            bits=UART.EIGHTBITS,
            parity=UART.PARITY_NONE,
            stop=UART.STOPBITS_ONE
        )

        print("[uart] UART%d ready, tx=IO%d, rx=IO%d, baud=%d, interval=%dms" %
              (uart_id, tx_pin, rx_pin, baudrate, send_interval_ms))

    def _pack_int16(self, val):
        """将 Python int 转为小端 int16 两个字节 (有符号)"""
        if val < -32768:
            val = -32768
        elif val > 32767:
            val = 32767
        lo = val & 0xFF
        hi = (val >> 8) & 0xFF
        return lo, hi

    def send(self, x_cm):
        """
        发送一帧 PnP x 偏移数据。
        :param x_cm: 水平偏移 (cm)
        :return: True 已发送, False 间隔不足跳过
        """
        now = time.ticks_ms()
        if time.ticks_diff(now, self._last_send) < self.send_interval_ms:
            return False
        self._last_send = now

        xi = int(x_cm * 100)

        frame = bytearray(7)
        frame[0] = 0xAA
        frame[1] = 0x55
        frame[2] = 0x02          # 数据长度: 2 bytes

        lo, hi = self._pack_int16(xi)
        frame[3] = lo
        frame[4] = hi

        # 校验和: XOR(byte2 .. byte4)
        cs = frame[2] ^ frame[3] ^ frame[4]
        frame[5] = cs & 0xFF

        frame[6] = 0xED          # 帧尾

        self.uart.write(bytes(frame))
        return True

    def deinit(self):
        """释放 UART 资源"""
        if self.uart:
            try:
                self.uart.deinit()
            except Exception:
                pass
            self.uart = None
        print("[uart] Deinit done")
