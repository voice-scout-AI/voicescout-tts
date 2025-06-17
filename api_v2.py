"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang: "",               # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path
    "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker tone fusion
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int. top k sampling
    "top_p": 1,                   # float. top p sampling
    "temperature": 1,             # float. temperature for sampling
    "text_split_method": "cut0",  # str. text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int. batch size for inference
    "batch_threshold": 0.75,      # float. threshold for batch splitting.
    "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float. control the speed of the synthesized audio.
    "streaming_mode": False,      # bool. whether to return a streaming response.
    "seed": -1,                   # int. random seed for reproducibility.
    "parallel_infer": True,       # bool. whether to use parallel inference.
    "repetition_penalty": 1.35    # float. repetition penalty for T2S model.
    "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
    "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP:
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400

"""

import os
import sys
import traceback
from typing import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
import sqlite3
from fastapi import FastAPI, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
from typing import List, Optional
import json
import yaml
from subprocess import Popen

# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
parser.add_argument("--init-db", action="store_true", help="데이터베이스를 초기화합니다 (테스트 데이터 포함)")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()

# SQLite 데이터베이스 초기화
def init_database():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    
    # users 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
    ''')
    
    # 테스트 데이터 삽입 (테이블이 비어있을 경우)
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        test_users = [
            ('김철수',),
            ('박영희',),
            ('이민수',),
            ('정수진',)
        ]
        cursor.executemany('INSERT INTO users (text) VALUES (?)', test_users)
    
    conn.commit()
    conn.close()

# 데이터베이스 초기화 (--init-db 인자가 있을 때만)
if args.init_db:
    print("데이터베이스를 초기화합니다...")
    init_database()
    print("데이터베이스 초기화가 완료되었습니다.")

class User(BaseModel):
    id: int
    text: str

class TrainRequest(BaseModel):
    exp_name: Optional[str] = "test"
    inp_text: Optional[str] = "GPT_SoVITS/output/asr_opt/slicer_opt.list"
    inp_wav_dir: Optional[str] = f"GPT_SoVITS/user/0"
    
# 훈련 상태 추적용 전역 변수
training_status = {
    "is_training": False,
    "current_stage": "",
    "progress": "",
    "error": None
}

class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    aux_ref_audio_paths: list = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 15
    top_p: float = 0.8
    temperature: float = 0.35
    text_split_method: str = "cut1"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = False
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
def pack_ogg(io_buffer: BytesIO, data: np.ndarray, rate: int):
    with sf.SoundFile(io_buffer, mode="w", samplerate=rate, channels=1, format="ogg") as audio_file:
        audio_file.write(data)
    return io_buffer


def pack_raw(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer.write(data.tobytes())
    return io_buffer


def pack_wav(io_buffer: BytesIO, data: np.ndarray, rate: int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format="wav")
    return io_buffer


def pack_aac(io_buffer: BytesIO, data: np.ndarray, rate: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-f",
            "s16le",  # 输入16位有符号小端整数PCM
            "-ar",
            str(rate),  # 设置采样率
            "-ac",
            "1",  # 单声道
            "-i",
            "pipe:0",  # 从管道读取输入
            "-c:a",
            "aac",  # 音频编码器为AAC
            "-b:a",
            "192k",  # 比特率
            "-vn",  # 不包含视频
            "-f",
            "adts",  # 输出AAC数据流格式
            "pipe:1",  # 将输出写入管道
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer


def pack_audio(io_buffer: BytesIO, data: np.ndarray, rate: int, media_type: str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer


# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def check_params(req: dict):
    text: str = req.get("text", "")
    text_lang: str = req.get("text_lang", "")
    ref_audio_path: str = req.get("ref_audio_path", "")
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if text_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"text_lang: {text_lang} is not supported in version {tts_config.version}"},
        )
    if prompt_lang in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(
            status_code=400,
            content={"message": f"prompt_lang: {prompt_lang} is not supported in version {tts_config.version}"},
        )
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": f"media_type: {media_type} is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})

    if text_split_method not in cut_method_names:
        return JSONResponse(
            status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"}
        )

    return None


async def tts_handle(req: dict):
    """
    Text to speech handler.

    Args:
        req (dict):
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "aux_ref_audio_paths": [],    # list.(optional) auxiliary reference audio paths for multi-speaker synthesis
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
                "sample_steps": 32,           # int. number of sampling steps for VITS model V3.
                "super_sampling": False,       # bool. whether to use super-sampling for audio when using VITS model V3.
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(tts_generator: Generator, media_type: str):
                if_frist_chunk = True
                for sr, chunk in tts_generator:
                    if if_frist_chunk and media_type == "wav":
                        yield wave_header_chunk(sample_rate=sr)
                        media_type = "raw"
                        if_frist_chunk = False
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()

            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                    media_type,
                ),
                media_type=f"audio/{media_type}",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)


@APP.get("/tts")
async def tts_get_endpoint(
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    aux_ref_audio_paths: list = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 15,
    top_p: float = 0.8,
    temperature: float = 0.35,
    text_split_method: str = "cut1",
    batch_size: int = 1,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = False,
    repetition_penalty: float = 1.35,
    sample_steps: int = 32,
    super_sampling: bool = False,
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "aux_ref_audio_paths": aux_ref_audio_paths,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": super_sampling,
    }
    return await tts_handle(req)


@APP.post("/tts")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)


@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})

#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())

#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/users", response_model=List[User])
async def get_users():
    """
    모든 사용자 목록을 조회합니다.
    
    Returns:
        List[User]: 사용자 목록 (id, text 포함)
    """
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, text FROM users ORDER BY id')
        users = cursor.fetchall()
        
        conn.close()
        
        # User 객체 리스트로 변환
        user_list = [User(id=user[0], text=user[1]) for user in users]
        
        return user_list
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": "데이터베이스 조회 실패", "Exception": str(e)})


def cleanup_experiment_files(exp_dir: str, tmp_dir: str, exp_name: str):
    """훈련 완료 후 불필요한 파일들 정리 및 최적 모델 파일명 변경"""
    import shutil
    import glob
    import re
    
    try:
        print(f"🗑️ 실험 파일 정리 및 모델 파일명 변경 시작: {exp_name}")
        
        # 1. 최적 모델 파일 이름 변경
        rename_best_model_files(exp_name)
        
        # 2. 실험 디렉토리 전체 삭제
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            print(f"✅ 삭제 완료: {exp_dir}")
        
        # 3. 임시 파일들 삭제
        temp_files = [
            f"{tmp_dir}/tmp_s2.json",
            f"{tmp_dir}/tmp_s1.yaml"
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"✅ 삭제 완료: {temp_file}")
        
        print(f"🎉 정리 완료! {exp_name} 최적 모델 파일들이 준비되었습니다.")
        
    except Exception as e:
        print(f"⚠️ 정리 중 오류: {e}")


def rename_best_model_files(exp_name: str):
    """가장 성능이 좋은 모델 파일들을 exp_name으로 이름 변경"""
    try:
        print(f"🔄 최적 모델 파일 이름 변경 중: {exp_name}")
        
        # SoVITS 모델 파일 처리
        sovits_dir = "SoVITS_weights_v2ProPlus"
        if os.path.exists(sovits_dir):
            # 가장 높은 에포크의 메인 모델 찾기
            sovits_pattern = f"{sovits_dir}/{exp_name}_e*_s*.pth"
            sovits_files = glob.glob(sovits_pattern)
            
            if sovits_files:
                # 에포크와 스텝 번호로 정렬해서 가장 최신 파일 선택
                def extract_epoch_step(filepath):
                    match = re.search(r'_e(\d+)_s(\d+)\.pth$', filepath)
                    if match:
                        return (int(match.group(1)), int(match.group(2)))
                    return (0, 0)
                
                best_sovits = max(sovits_files, key=extract_epoch_step)
                new_sovits_name = f"{sovits_dir}/{exp_name}.pth"
                
                if os.path.exists(new_sovits_name):
                    os.remove(new_sovits_name)
                os.rename(best_sovits, new_sovits_name)
                print(f"✅ SoVITS 모델: {best_sovits} → {new_sovits_name}")
                
                # LoRA 파일들은 TTS 추론에 불필요하므로 삭제
                lora_pattern = f"{sovits_dir}/{exp_name}_e*_s*_lora.ckpt"
                lora_files = glob.glob(lora_pattern)
                for lora_file in lora_files:
                    os.remove(lora_file)
                    print(f"🗑️ LoRA 파일 삭제 (추론에 불필요): {lora_file}")
        
        # GPT 모델 파일 처리
        gpt_dir = "GPT_weights_v2ProPlus"
        if os.path.exists(gpt_dir):
            # 가장 높은 에포크의 모델 찾기
            gpt_pattern = f"{gpt_dir}/{exp_name}-e*.ckpt"
            gpt_files = glob.glob(gpt_pattern)
            
            if gpt_files:
                # 에포크 번호로 정렬해서 가장 최신 파일 선택
                def extract_gpt_epoch(filepath):
                    match = re.search(r'-e(\d+)\.ckpt$', filepath)
                    if match:
                        return int(match.group(1))
                    return 0
                
                best_gpt = max(gpt_files, key=extract_gpt_epoch)
                new_gpt_name = f"{gpt_dir}/{exp_name}.ckpt"
                
                if os.path.exists(new_gpt_name):
                    os.remove(new_gpt_name)
                os.rename(best_gpt, new_gpt_name)
                print(f"✅ GPT 모델: {best_gpt} → {new_gpt_name}")
        
        # 중간 체크포인트 파일들 정리 (선택사항)
        cleanup_intermediate_checkpoints(exp_name)
        
    except Exception as e:
        print(f"⚠️ 모델 파일 이름 변경 중 오류: {e}")


def cleanup_intermediate_checkpoints(exp_name: str):
    """중간 체크포인트 파일들 정리 (최종 모델만 남기고 삭제)"""
    try:
        # SoVITS 중간 파일들 삭제
        sovits_dir = "SoVITS_weights_v2ProPlus"
        if os.path.exists(sovits_dir):
            for file in os.listdir(sovits_dir):
                if (exp_name in file and 
                    (file.endswith('.pth') or file.endswith('.ckpt')) and
                    file != f"{exp_name}.pth"):  # 메인 모델 파일만 보존
                    file_path = f"{sovits_dir}/{file}"
                    os.remove(file_path)
                    print(f"🗑️ 중간 파일 삭제: {file_path}")
        
        # GPT 중간 파일들 삭제
        gpt_dir = "GPT_weights_v2ProPlus"
        if os.path.exists(gpt_dir):
            for file in os.listdir(gpt_dir):
                if (exp_name in file and 
                    file.endswith('.ckpt') and
                    file != f"{exp_name}.ckpt"):
                    file_path = f"{gpt_dir}/{file}"
                    os.remove(file_path)
                    print(f"🗑️ 중간 파일 삭제: {file_path}")
                    
        print("✅ 중간 체크포인트 파일들 정리 완료")
        
    except Exception as e:
        print(f"⚠️ 중간 파일 정리 중 오류: {e}")


def run_training_pipeline(request: TrainRequest):
    """백그라운드에서 실행되는 훈련 파이프라인"""
    global training_status
    
    try:
        training_status.update({
            "is_training": True,
            "current_stage": "시작",
            "progress": "훈련 준비 중...",
            "error": None
        })
        
        # 현재 디렉토리 및 기본 설정
        now_dir = os.getcwd()
        tmp_dir = os.path.join(now_dir, "TEMP")
        os.makedirs(tmp_dir, exist_ok=True)
        
        # 실험명 일관성 확보
        exp_name = request.exp_name.rstrip(' ')
        exp_root = "logs"  # 실험 디렉토리 루트
        exp_dir = f"{exp_root}/{exp_name}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # Python 실행자 설정
        python_exec = sys.executable or "python"
        
        # 1. 데이터셋 형식화 (1Aabc)
        training_status.update({
            "current_stage": "데이터셋 형식화",
            "progress": "1A - 텍스트 분할 및 특징 추출 중..."
        })
        
        # 1a: 텍스트 분할 및 특징 추출
        path_text = f"{exp_dir}/2-name2text.txt"
        if not os.path.exists(path_text) or (
            os.path.exists(path_text) and 
            len(open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2
        ):
            config = {
                "inp_text": request.inp_text,
                "inp_wav_dir": request.inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": exp_dir,
                "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                "is_half": "True",
                "i_part": "0",
                "all_parts": "1",
                "_CUDA_VISIBLE_DEVICES": "0"
            }
            
            for key, value in config.items():
                os.environ[key] = str(value)
            
            cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/1-get-text.py'
            print(f"실행 명령어: {cmd}")
            process = Popen(cmd, shell=True)
            process.wait()
            
            # 개별 파트 파일들을 하나로 병합 (webui.py 로직과 동일)
            opt = []
            i_part = 0
            while True:
                part_file = f"{exp_dir}/2-name2text-{i_part}.txt"
                if not os.path.exists(part_file):
                    break
                print(f"파트 파일 병합 중: {part_file}")
                with open(part_file, "r", encoding="utf8") as f:
                    opt += f.read().strip("\n").split("\n")
                os.remove(part_file)  # 개별 파일 삭제
                i_part += 1
            
            # 통합 파일 생성
            if opt:
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print(f"병합 완료: {len(opt)}개 항목을 {path_text}에 저장")
            
            # 결과 검증
            if not os.path.exists(path_text) or os.path.getsize(path_text) == 0:
                raise Exception("1A 단계 실패: 텍스트 파일이 생성되지 않았습니다.")
        
        # 1b: 음성 자기지도 특징 추출
        training_status.update({
            "current_stage": "데이터셋 형식화",
            "progress": "1B - 음성 자기지도 특징 추출 중..."
        })
        
        config.update({
            "cnhubert_base_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "sv_path": "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        })
        
        for key, value in config.items():
            os.environ[key] = str(value)
            
        cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'
        print(f"실행 명령어: {cmd}")
        process = Popen(cmd, shell=True)
        process.wait()
        
        cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py'
        print(f"실행 명령어: {cmd}")
        process = Popen(cmd, shell=True)
        process.wait()
        
        # 1c: 의미론적 토큰 추출
        training_status.update({
            "current_stage": "데이터셋 형식화",
            "progress": "1C - 의미론적 토큰 추출 중..."
        })
        
        path_semantic = f"{exp_dir}/6-name2semantic.tsv"
        if not os.path.exists(path_semantic) or os.path.getsize(path_semantic) < 31:
            config.update({
                "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
                "s2config_path": "GPT_SoVITS/configs/s2v2ProPlus.json"
            })
            
            for key, value in config.items():
                os.environ[key] = str(value)
                
            cmd = f'"{python_exec}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py'
            print(f"실행 명령어: {cmd}")
            process = Popen(cmd, shell=True)
            process.wait()
            
            # 개별 파트 파일들을 하나로 병합 (webui.py 로직과 동일)
            opt = ["item_name\tsemantic_audio"]  # 헤더 추가
            i_part = 0
            while True:
                part_file = f"{exp_dir}/6-name2semantic-{i_part}.tsv"
                if not os.path.exists(part_file):
                    break
                print(f"파트 파일 병합 중: {part_file}")
                with open(part_file, "r", encoding="utf8") as f:
                    opt += f.read().strip("\n").split("\n")
                os.remove(part_file)  # 개별 파일 삭제
                i_part += 1
            
            # 통합 파일 생성
            if len(opt) > 1:  # 헤더 외에 데이터가 있는 경우
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print(f"병합 완료: {len(opt)-1}개 항목을 {path_semantic}에 저장")
            
            # 결과 검증
            if not os.path.exists(path_semantic) or os.path.getsize(path_semantic) < 31:
                raise Exception("1C 단계 실패: 의미론적 토큰 파일이 생성되지 않았습니다.")
        
        # 2. SoVITS 훈련
        training_status.update({
            "current_stage": "SoVITS 훈련",
            "progress": "SoVITS 모델 훈련 중..."
        })
        
        # SoVITS 설정 파일 생성 (v2ProPlus 고정)
        with open("GPT_SoVITS/configs/s2v2ProPlus.json") as f:
            s2_config = json.load(f)
        
        # 로그 디렉토리 생성
        os.makedirs(f"{exp_dir}/logs_s2_v2ProPlus", exist_ok=True)
        
        # 설정 업데이트
        s2_config["train"]["batch_size"] = 7
        s2_config["train"]["epochs"] = 8
        s2_config["train"]["text_low_lr_rate"] = 0.4
        s2_config["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
        s2_config["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth".replace("s2G", "s2D")
        s2_config["train"]["if_save_latest"] = True
        s2_config["train"]["if_save_every_weights"] = True
        s2_config["train"]["save_every_epoch"] = 4
        s2_config["train"]["gpu_numbers"] = "0"
        s2_config["train"]["grad_ckpt"] = False
        s2_config["train"]["lora_rank"] = "32"
        s2_config["model"]["version"] = "v2ProPlus"
        s2_config["data"]["exp_dir"] = exp_dir
        s2_config["s2_ckpt_dir"] = exp_dir
        s2_config["save_weight_dir"] = "SoVITS_weights_v2ProPlus"
        s2_config["name"] = exp_name
        s2_config["version"] = "v2ProPlus"
        
        tmp_s2_config = f"{tmp_dir}/tmp_s2.json"
        with open(tmp_s2_config, "w") as f:
            json.dump(s2_config, f)
        
        cmd = f'"{python_exec}" -s GPT_SoVITS/s2_train.py --config "{tmp_s2_config}"'
            
        print(f"실행 명령어: {cmd}")
        process = Popen(cmd, shell=True)
        process.wait()
        
        # 3. GPT 훈련
        training_status.update({
            "current_stage": "GPT 훈련",
            "progress": "GPT 모델 훈련 중..."
        })
        
        # GPT 설정 파일 생성 (v2ProPlus 고정)
        with open("GPT_SoVITS/configs/s1longer-v2.yaml") as f:
            s1_config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 로그 디렉토리 생성
        os.makedirs(f"{exp_dir}/logs_s1", exist_ok=True)
        
        # 설정 업데이트
        s1_config["train"]["batch_size"] = 7
        s1_config["train"]["epochs"] = 15
        s1_config["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        s1_config["train"]["save_every_n_epoch"] = 5
        s1_config["train"]["if_save_every_weights"] = True
        s1_config["train"]["if_save_latest"] = True
        s1_config["train"]["if_dpo"] = False
        s1_config["train"]["half_weights_save_dir"] = "GPT_weights_v2ProPlus"
        s1_config["train"]["exp_name"] = exp_name
        s1_config["train_semantic_path"] = f"{exp_dir}/6-name2semantic.tsv"
        s1_config["train_phoneme_path"] = f"{exp_dir}/2-name2text.txt"
        s1_config["output_dir"] = f"{exp_dir}/logs_s1_v2ProPlus"
        
        tmp_s1_config = f"{tmp_dir}/tmp_s1.yaml"
        with open(tmp_s1_config, "w") as f:
            yaml.dump(s1_config, f, default_flow_style=False)
        
        # 환경변수 설정
        os.environ["_CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["hz"] = "25hz"
        
        cmd = f'"{python_exec}" -s GPT_SoVITS/s1_train.py --config_file "{tmp_s1_config}"'
        print(f"실행 명령어: {cmd}")
        process = Popen(cmd, shell=True)
        process.wait()
        
        # 훈련 완료 후 정리 작업 (선택사항)
        training_status.update({
            "current_stage": "정리 작업",
            "progress": "훈련 파일 정리 중..."
        })
        
        # 실험 디렉토리 정리 (주석 해제하면 자동 삭제)
        cleanup_experiment_files(exp_dir, tmp_dir, exp_name)
        
        # 훈련 완료
        training_status.update({
            "is_training": False,
            "current_stage": "완료", 
            "progress": "모든 훈련이 성공적으로 완료되었습니다!",
            "error": None
        })
        
    except Exception as e:
        training_status.update({
            "is_training": False,
            "current_stage": "오류",
            "progress": "",
            "error": str(e)
        })
        print(f"훈련 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


@APP.post("/train")
async def start_training(request: TrainRequest, background_tasks: BackgroundTasks):
    """
    GPT-SoVITS 모델 훈련을 시작합니다.
    1Aabc 원클릭 형식화 -> SoVITS 훈련 -> GPT 훈련 순서로 진행됩니다.
    """
    global training_status
    
    if training_status["is_training"]:
        return JSONResponse(
            status_code=400, 
            content={"message": "이미 훈련이 진행 중입니다. /train/status로 상태를 확인하세요."}
        )
    
    # 필수 파라미터 검증
    if not request.exp_name or not request.inp_text or not request.inp_wav_dir:
        return JSONResponse(
            status_code=400,
            content={"message": "exp_name, inp_text, inp_wav_dir은 필수 파라미터입니다."}
        )
    
    # 백그라운드 태스크로 훈련 시작
    background_tasks.add_task(run_training_pipeline, request)
    
    return JSONResponse(
        status_code=200,
        content={
            "message": "훈련이 백그라운드에서 시작되었습니다.",
            "exp_name": request.exp_name,
            "status_check_url": "/train/status"
        }
    )


@APP.get("/train/status")
async def get_training_status():
    """현재 훈련 상태를 조회합니다."""
    return JSONResponse(status_code=200, content=training_status)


@APP.post("/train/stop")
async def stop_training():
    """진행 중인 훈련을 중단합니다."""
    global training_status
    
    if not training_status["is_training"]:
        return JSONResponse(
            status_code=400,
            content={"message": "현재 진행 중인 훈련이 없습니다."}
        )
    
    # 간단한 상태 리셋 (실제 프로세스 종료는 복잡하므로 상태만 변경)
    training_status.update({
        "is_training": False,
        "current_stage": "중단됨",
        "progress": "사용자에 의해 훈련이 중단되었습니다.",
        "error": None
    })
    
    return JSONResponse(
        status_code=200,
        content={"message": "훈련 중단 요청이 처리되었습니다."}
    )


if __name__ == "__main__":
    try:
        if host == "None":  # 在调用时使用 -a None 参数，可以让api监听双栈
            host = None
        uvicorn.run(app=APP, host=host, port=port)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
