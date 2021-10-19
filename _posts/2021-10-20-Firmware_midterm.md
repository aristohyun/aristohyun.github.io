---
layout: post
title: "Firmware, midterms"
description: "Firmware, 전중남 교수님"
categories: [DBS]
tags: [2021-2, Firmware, 전중남]
use_math: true
redirect_from:
  - /2021/10/20/
---

* Kramdown table of contents
{:toc .toc}   

컴퓨터 시스템 = 하드웨어(중앙처리장치,기억장치, 입출력장치) + 소프트웨어
임베디드시스템 : 다른 장비에 포함되어있는 컴퓨터시스템, 제어기능 수행
펌웨어: 소형 임베디드 시스템, ROM에 저장되어 실행되는 프로그램

원래 롬에 저장되어서 수정이 불가능했음. 그러나 재프로그램이 가능한 programmable ROM[^programmable_ROM]에 기록함
임의수정 X, 재프로그래밍(덮어씌우기)

ROM : non vaolatile, 전원이 제거되더라도 내용이 유지
RAM : vaolatile, 휘발성, 전원이 제거되면 내용이 삭제

[^programmable_ROM]: 재프로그램이 가능한 형태의 ROM,   EPROM, Flash memory 등


MCU
CPU, 메모리 ,IO를 하나의 칩에 구현한 반도체 소자
소규모 임베디드용, 펌웨어 프로그래밍 대상

MPU
보조기억장치(플래시 메모리 탑재)
운영체제를 탑재하기 위해서 프로세서는 가상 기억장치를 제어하기 위한 기억장치 관리장치 MMU를 제공해야 함

운ㅇㅇ체제를 탑재했다 ? 시스템 콜, 운영체제 기능, 라이브러리 함수 사용가능
파일시스템 디바이스 드라이버 TCP/IP 네트워크 통신

그러나 펌웨어프로그래밍에서는 사용 불가

AVR 8비트 마이크로제어기
- 3가지 제품군 ATiny, ATmege, Xmega

펌웨어 개발 환경

임베디드 시스템 개발환경

