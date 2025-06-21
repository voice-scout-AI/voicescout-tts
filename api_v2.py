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
from fastapi import FastAPI, Response, BackgroundTasks, UploadFile, File, Form
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
    # 기존 데이터베이스 파일이 있으면 삭제
    if os.path.exists('users.db'):
        os.remove('users.db')
        print("🗑️ 기존 데이터베이스 파일 삭제: users.db")
    
    # 새로운 데이터베이스 생성 및 연결
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    print("📝 새로운 데이터베이스 파일 생성: users.db")
    
    # users 테이블 생성
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL
        )
    ''')
    print("📋 users 테이블 생성 완료")
    
    # 테스트 데이터 삽입
    test_users = [
        ('ShinJjang',),
        ('MoDongSoop',),
        ('DrPark',),
        ('EJKim',),
    ]
    cursor.executemany('INSERT INTO users (text) VALUES (?)', test_users)
    print(f"✅ 테스트 데이터 {len(test_users)}개 삽입 완료")
    
    conn.commit()
    conn.close()
    print("🎉 데이터베이스 초기화 완료!")

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
    exp_name: str = "test"


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
    streaming_mode: bool = req.get("streaming_mode", False)
    media_type: str = req.get("media_type", "wav")
    prompt_lang: str = req.get("prompt_lang", "")
    text_split_method: str = req.get("text_split_method", "cut5")

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
                "exp_name": "",               # str.(required) experiment name to auto-load checkpoints and reference audio from GPT_SoVITS/user/{exp_name}/voice0.wav
            }
    returns:
        StreamingResponse: audio stream response.
    """

    streaming_mode = req.get("streaming_mode", False)
    return_fragment = req.get("return_fragment", False)
    media_type = req.get("media_type", "wav")
    exp_name = req.get("exp_name", "test")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    # exp_name이 제공된 경우 해당 실험의 체크포인트 및 참조 음성을 자동으로 로드
    if exp_name and exp_name.strip():
        try:
            user_model_dir = f"GPT_SoVITS/user/{exp_name.strip()}"
            gpt_path = f"{user_model_dir}/{exp_name.strip()}.ckpt"
            sovits_path = f"{user_model_dir}/{exp_name.strip()}.pth"
            ref_audio_path = f"{user_model_dir}/voice0.wav"
            
            # 참조 음성 파일 존재 확인
            if not os.path.exists(ref_audio_path):
                return JSONResponse(
                    status_code=400, 
                    content={"message": f"참조 음성 파일을 찾을 수 없습니다: {ref_audio_path}"}
                )
            
            # GPT 모델 체크포인트 존재 확인 및 로드
            if os.path.exists(gpt_path):
                print(f"🔄 GPT 모델 로드 중: {gpt_path}")
                tts_pipeline.init_t2s_weights(gpt_path)
            else:
                return JSONResponse(
                    status_code=400, 
                    content={"message": f"GPT 체크포인트를 찾을 수 없습니다: {gpt_path}"}
                )
            
            # SoVITS 모델 체크포인트 존재 확인 및 로드
            if os.path.exists(sovits_path):
                print(f"🔄 SoVITS 모델 로드 중: {sovits_path}")
                tts_pipeline.init_vits_weights(sovits_path)
            else:
                return JSONResponse(
                    status_code=400, 
                    content={"message": f"SoVITS 체크포인트를 찾을 수 없습니다: {sovits_path}"}
                )
            
            # req에 참조 음성 경로 자동 설정
            req["ref_audio_path"] = ref_audio_path
            print(f"🎤 참조 음성 설정: {ref_audio_path}")
            print(f"✅ {exp_name} 실험의 모델들이 성공적으로 로드되었습니다.")
            
        except Exception as e:
            return JSONResponse(
                status_code=400, 
                content={"message": f"모델 로드 실패", "Exception": str(e)}
            )
    else:
        return JSONResponse(
            status_code=400,
            content={"message": "exp_name은 필수 파라미터입니다."}
        )

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
    exp_name: str = "test",
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
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
        "exp_name": exp_name,
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


@APP.delete("/users/{user_id}")
async def delete_user_by_id(user_id: int):
    """
    ID로 사용자를 삭제합니다. (데이터베이스에서만 삭제, 파일은 보존)
    
    Args:
        user_id (int): 삭제할 사용자의 ID
        
    Returns:
        성공/실패 메시지
    """
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # 먼저 사용자가 존재하는지 확인
        cursor.execute('SELECT id, text FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return JSONResponse(
                status_code=404, 
                content={"message": f"ID {user_id}에 해당하는 사용자를 찾을 수 없습니다."}
            )
        
        user_text = user[1]
        
        # 사용자 삭제
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "사용자가 성공적으로 삭제되었습니다.",
                    "deleted_user": {
                        "id": user_id,
                        "text": user_text
                    },
                    "note": f"파일은 보존됩니다: GPT_SoVITS/user/{user_text}/"
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"message": "사용자 삭제에 실패했습니다."}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"message": "사용자 삭제 중 오류가 발생했습니다.", "Exception": str(e)}
        )


@APP.delete("/users")
async def delete_user_by_text(text: str):
    """
    텍스트(exp_name)로 사용자를 삭제합니다. (데이터베이스에서만 삭제, 파일은 보존)
    
    Args:
        text (str): 삭제할 사용자의 텍스트(실험명)
        
    Returns:
        성공/실패 메시지
    """
    try:
        if not text or not text.strip():
            return JSONResponse(
                status_code=400,
                content={"message": "text 파라미터는 필수입니다."}
            )
        
        text = text.strip()
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # 먼저 사용자가 존재하는지 확인
        cursor.execute('SELECT id, text FROM users WHERE text = ?', (text,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return JSONResponse(
                status_code=404, 
                content={"message": f"'{text}'에 해당하는 사용자를 찾을 수 없습니다."}
            )
        
        user_id = user[0]
        
        # 사용자 삭제
        cursor.execute('DELETE FROM users WHERE text = ?', (text,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        if deleted_count > 0:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "사용자가 성공적으로 삭제되었습니다.",
                    "deleted_user": {
                        "id": user_id,
                        "text": text
                    },
                    "note": f"파일은 보존됩니다: GPT_SoVITS/user/{text}/"
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"message": "사용자 삭제에 실패했습니다."}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"message": "사용자 삭제 중 오류가 발생했습니다.", "Exception": str(e)}
        )


def cleanup_experiment_files(exp_dir: str, tmp_dir: str, exp_name: str):
    """훈련 완료 후 최적 모델 선별 및 GPT_SoVITS/user/{exp_name}/ 폴더에 저장"""
    import shutil
    import glob
    import re
    
    try:
        print(f"🎯 최적 모델 선별 및 저장 시작: {exp_name}")
        
        # 1. 최적 모델 선별 및 저장
        save_best_models_to_user_folder(exp_name, exp_dir)
        
        # 2. 모든 실험 파일 정리 (logs 디렉토리)
        cleanup_all_experiment_files(exp_dir)
        
        # 3. 임시 파일들 삭제
        cleanup_temp_files(tmp_dir)
        
        print(f"🎉 완료! {exp_name} 최적 모델이 GPT_SoVITS/user/{exp_name}/에 저장되었습니다.")
        
    except Exception as e:
        print(f"⚠️ 정리 중 오류: {e}")


def save_best_models_to_user_folder(exp_name: str, exp_dir: str):
    """최적 모델들을 GPT_SoVITS/user/{exp_name}/ 폴더에 저장"""
    import shutil
    import glob
    import re
    
    try:
        # 사용자 폴더 생성
        user_folder = f"GPT_SoVITS/user/{exp_name}"
        os.makedirs(user_folder, exist_ok=True)
        print(f"📁 사용자 폴더 생성: {user_folder}")
        
        # 1. SoVITS 최적 모델 찾기 및 저장
        sovits_dir = f"{exp_dir}/SoVITS_weights"
        if os.path.exists(sovits_dir):
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
                final_sovits_path = f"{user_folder}/{exp_name}.pth"
                
                shutil.copy2(best_sovits, final_sovits_path)
                print(f"✅ SoVITS 최적 모델 저장: {best_sovits} → {final_sovits_path}")
            else:
                print(f"⚠️ SoVITS 훈련 모델을 찾을 수 없습니다: {sovits_pattern}")
        
        # 2. GPT 최적 모델 찾기 및 저장
        gpt_dir = f"{exp_dir}/GPT_weights"
        if os.path.exists(gpt_dir):
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
                final_gpt_path = f"{user_folder}/{exp_name}.ckpt"
                
                shutil.copy2(best_gpt, final_gpt_path)
                print(f"✅ GPT 최적 모델 저장: {best_gpt} → {final_gpt_path}")
            else:
                print(f"⚠️ GPT 훈련 모델을 찾을 수 없습니다: {gpt_pattern}")
        
        # 3. 루트 디렉토리에서도 모델 찾기 (기존 훈련된 모델이 있을 경우)
        root_gpt_pattern = f"GPT_weights_v2ProPlus/{exp_name}-e*.ckpt"
        root_gpt_files = glob.glob(root_gpt_pattern)
        
        if root_gpt_files and not gpt_files:  # exp_dir에 없고 루트에만 있는 경우
            def extract_gpt_epoch(filepath):
                match = re.search(r'-e(\d+)\.ckpt$', filepath)
                if match:
                    return int(match.group(1))
                return 0
            
            best_root_gpt = max(root_gpt_files, key=extract_gpt_epoch)
            final_gpt_path = f"{user_folder}/{exp_name}.ckpt"
            
            shutil.copy2(best_root_gpt, final_gpt_path)
            print(f"✅ 루트 GPT 모델 복사: {best_root_gpt} → {final_gpt_path}")
            
    except Exception as e:
        print(f"⚠️ 최적 모델 저장 중 오류: {e}")


def cleanup_all_experiment_files(exp_dir: str):
    """실험 디렉토리 전체 삭제"""
    import shutil
    
    try:
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
            print(f"🗑️ 실험 디렉토리 삭제: {exp_dir}")
        
        # 루트 디렉토리의 모든 중간 모델들도 정리
        cleanup_root_intermediate_files()
            
    except Exception as e:
        print(f"⚠️ 실험 파일 삭제 중 오류: {e}")


def cleanup_root_intermediate_files():
    """루트 디렉토리의 중간 훈련 파일들 정리"""
    import glob
    
    try:
        # GPT 중간 모델들 삭제 (exp_name-e*.ckpt 형태)
        gpt_files = glob.glob("GPT_weights_v2ProPlus/*-e*.ckpt")
        for gpt_file in gpt_files:
            os.remove(gpt_file)
            print(f"🗑️ GPT 중간 모델 삭제: {gpt_file}")
        
        # SoVITS 중간 모델들 삭제 (exp_name_e*_s*.pth 형태)
        sovits_files = glob.glob("SoVITS_weights_v2ProPlus/*_e*_s*.pth")
        for sovits_file in sovits_files:
            os.remove(sovits_file)
            print(f"🗑️ SoVITS 중간 모델 삭제: {sovits_file}")
        
        # LoRA 파일들 삭제
        lora_files = glob.glob("SoVITS_weights_v2ProPlus/*_lora.ckpt")
        for lora_file in lora_files:
            os.remove(lora_file)
            print(f"🗑️ LoRA 파일 삭제: {lora_file}")
            
    except Exception as e:
        print(f"⚠️ 루트 중간 파일 정리 중 오류: {e}")


def cleanup_temp_files(tmp_dir: str):
    """임시 파일들 삭제"""
    try:
        temp_files = [
            f"{tmp_dir}/tmp_s2.json",
            f"{tmp_dir}/tmp_s1.yaml"
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"✅ 임시 파일 삭제: {temp_file}")
                
    except Exception as e:
        print(f"⚠️ 임시 파일 삭제 중 오류: {e}")


def run_training_pipeline(request: TrainRequest):
    """백그라운드에서 실행되는 훈련 파이프라인"""
    global training_status
    
    # 변수 초기화 (finally 블록에서 사용하기 위해)
    exp_name = None
    exp_dir = None
    tmp_dir = None
    
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
        
        # 훈련에 필요한 모든 디렉토리 미리 생성
        required_dirs = [
            f"{exp_dir}/SoVITS_weights",
            f"{exp_dir}/GPT_weights", 
            f"{exp_dir}/logs_s1",
            f"{exp_dir}/logs_s2_v2ProPlus",
            f"GPT_SoVITS/user/{exp_name}"  # 입력 음성 파일 디렉토리도 생성
        ]
        for dir_path in required_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 디렉토리 생성: {dir_path}")
        
        # 내부적으로 경로 설정
        inp_text = f"GPT_SoVITS/user/{exp_name}/slicer_opt.list"  # 고정 기본값 사용
        inp_wav_dir = f"GPT_SoVITS/user/{exp_name}"  # exp_name에 따라 자동 설정
        
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
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
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
        
        # 로그 및 모델 저장 디렉토리 생성
        os.makedirs(f"{exp_dir}/logs_s2_v2ProPlus", exist_ok=True)
        os.makedirs(f"{exp_dir}/SoVITS_weights", exist_ok=True)
        
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
        s2_config["save_weight_dir"] = f"{exp_dir}/SoVITS_weights"  # 실험 디렉토리 안으로 통합
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
        
        # 로그 및 모델 저장 디렉토리 생성  
        os.makedirs(f"{exp_dir}/logs_s1", exist_ok=True)
        os.makedirs(f"{exp_dir}/GPT_weights", exist_ok=True)
        
        # 설정 업데이트
        s1_config["train"]["batch_size"] = 7
        s1_config["train"]["epochs"] = 15
        s1_config["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        s1_config["train"]["save_every_n_epoch"] = 5
        s1_config["train"]["if_save_every_weights"] = True
        s1_config["train"]["if_save_latest"] = True
        s1_config["train"]["if_dpo"] = False
        s1_config["train"]["half_weights_save_dir"] = f"{exp_dir}/GPT_weights"  # 실험 디렉토리 안으로 통합
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
        
        # 훈련 완료 - 데이터베이스에 exp_name 추가
        try:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            
            # 중복 체크
            cursor.execute('SELECT COUNT(*) FROM users WHERE text = ?', (exp_name,))
            if cursor.fetchone()[0] == 0:
                # 중복되지 않는 경우에만 추가
                cursor.execute('INSERT INTO users (text) VALUES (?)', (exp_name,))
                conn.commit()
                print(f"✅ 훈련 완료 후 데이터베이스에 exp_name 추가 완료: {exp_name}")
            else:
                print(f"ℹ️ exp_name이 이미 데이터베이스에 존재함: {exp_name}")
            
            conn.close()
            
        except Exception as db_error:
            print(f"⚠️ 데이터베이스 추가 실패 (훈련은 성공): {db_error}")
        
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
        
    finally:
        # 성공/실패 여부와 관계없이 파일 정리 작업 진행
        if exp_name and exp_dir and tmp_dir:
            try:
                training_status.update({
                    "current_stage": "정리 작업",
                    "progress": "훈련 파일 정리 중..."
                })
                print(f"🧹 파일 정리 작업 시작 (성공/실패 무관): {exp_name}")
                cleanup_experiment_files(exp_dir, tmp_dir, exp_name)
            except Exception as cleanup_error:
                print(f"⚠️ 파일 정리 중 오류 (무시하고 계속): {cleanup_error}")
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
    if not request.exp_name:
        return JSONResponse(
            status_code=400,
            content={"message": "exp_name은 필수 파라미터입니다."}
        )
    
    # 백그라운드 태스크로 훈련 시작
    background_tasks.add_task(run_training_pipeline, request)
    
    return JSONResponse(
        status_code=200,
        content={
            "message": "훈련이 백그라운드에서 시작되었습니다.",
            "exp_name": request.exp_name,
            "inp_text": f"GPT_SoVITS/user/{request.exp_name}/slicer_opt.list",
            "inp_wav_dir": f"GPT_SoVITS/user/{request.exp_name}",
            "output_folder": f"GPT_SoVITS/user/{request.exp_name}",
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


@APP.post("/upload/voice")
async def upload_voice_files(
    exp_name: str = Form(...),
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    음성 파일들을 업로드하고 WAV 형식으로 변환한 후 자동으로 훈련을 시작합니다.
    
    Args:
        exp_name: 실험명 (폴더명으로 사용됨)
        files: 업로드할 음성/비디오 파일들 
               (MP4, AVI, MOV, MKV, FLV, MP3, M4A, AAC, OGG, WAV 지원)
               MP4 등 비디오 파일은 자동으로 WAV로 변환됩니다.
        background_tasks: FastAPI BackgroundTasks (자동 훈련 시작용)
    
    Returns:
        업로드, 변환 및 훈련 시작 결과 메시지
        - MP4/비디오 파일: 자동으로 WAV로 변환 (22050Hz, 모노)
        - WAV 파일: 그대로 저장
        - 기타 오디오 파일: WAV로 변환
        - 업로드 성공 시 자동으로 GPT-SoVITS 훈련 시작
    """
    try:
        # exp_name 검증
        if not exp_name or not exp_name.strip():
            return JSONResponse(
                status_code=400,
                content={"message": "exp_name은 필수 파라미터입니다."}
            )
        
        exp_name = exp_name.strip()
        
        # 업로드 디렉토리 생성
        upload_dir = f"GPT_SoVITS/user/{exp_name}"
        os.makedirs(upload_dir, exist_ok=True)
        
        # 허용된 파일명 목록
        allowed_filenames = [f"voice{i}.wav" for i in range(10)]  # voice0.wav ~ voice9.wav
        
        uploaded_files = []
        errors = []
        
        for file in files:
            try:
                # 파일 크기 검증 (100MB 제한)
                content = await file.read()
                if len(content) > 100 * 1024 * 1024:  # 100MB
                    errors.append(f"파일 '{file.filename}'의 크기가 너무 큽니다 (최대 100MB).")
                    continue
                
                # 파일 확장자 확인
                original_filename = file.filename
                file_extension = original_filename.lower().split('.')[-1] if '.' in original_filename else ''
                
                # 지원되는 형식인지 확인 (audio, video 모두 허용)
                supported_extensions = ['wav', 'mp3', 'mp4', 'avi', 'mov', 'mkv', 'flv', 'm4a', 'aac', 'ogg']
                if file_extension not in supported_extensions:
                    errors.append(f"파일 '{original_filename}'의 형식이 지원되지 않습니다. 지원 형식: {', '.join(supported_extensions)}")
                    continue
                
                # 임시 파일로 저장
                temp_input_path = os.path.join(upload_dir, f"temp_{original_filename}")
                with open(temp_input_path, "wb") as f:
                    f.write(content)
                
                # 최종 저장 파일명 생성 (확장자를 .wav로 변경)
                final_filename = original_filename.rsplit('.', 1)[0] + '.wav' if '.' in original_filename else original_filename + '.wav'
                final_file_path = os.path.join(upload_dir, final_filename)
                
                # WAV가 아닌 경우 변환, WAV인 경우 그대로 복사
                if file_extension != 'wav':
                    print(f"🔄 {file_extension.upper()} → WAV 변환 시작: {original_filename}")
                    
                    # ffmpeg를 사용해서 WAV로 변환
                    ffmpeg_cmd = [
                        "ffmpeg",
                        "-i", temp_input_path,     # 입력 파일
                        "-acodec", "pcm_s16le",    # 오디오 코덱: 16비트 PCM
                        "-ar", "22050",            # 샘플링 레이트: 22050Hz (TTS에 적합)
                        "-ac", "1",                # 모노 채널
                        "-y",                      # 출력 파일 덮어쓰기
                        final_file_path            # 출력 파일
                    ]
                    
                    process = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True,
                        timeout=60  # 60초 타임아웃
                    )
                    
                    if process.returncode != 0:
                        errors.append(f"파일 '{original_filename}' 변환 실패: {process.stderr}")
                        # 임시 파일 정리
                        if os.path.exists(temp_input_path):
                            os.remove(temp_input_path)
                        continue
                    
                    print(f"✅ 변환 완료: {original_filename} → {final_filename}")
                else:
                    # WAV 파일인 경우 그대로 복사
                    import shutil
                    shutil.move(temp_input_path, final_file_path)
                    print(f"✅ WAV 파일 저장: {final_filename}")
                
                # 임시 파일 정리
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                
                uploaded_files.append(final_filename)
                print(f"✅ 파일 처리 완료: {final_file_path}")
                
            except subprocess.TimeoutExpired:
                errors.append(f"파일 '{file.filename}' 변환 시간 초과 (60초)")
                # 임시 파일 정리
                temp_path = os.path.join(upload_dir, f"temp_{file.filename}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue
            except Exception as e:
                errors.append(f"파일 '{file.filename}' 처리 중 오류: {str(e)}")
                # 임시 파일 정리
                temp_path = os.path.join(upload_dir, f"temp_{file.filename}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue
        
        # slicer_opt.list 파일 생성
        try:
            template_path = "GPT_SoVITS/user/slicer_opt.list"
            target_path = f"{upload_dir}/slicer_opt.list"
            
            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    template_content = f.read()
                
                # {} 부분을 exp_name으로 치환
                updated_content = template_content.replace("{}", exp_name)
                
                # 새로운 slicer_opt.list 파일 생성
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                
                print(f"✅ slicer_opt.list 파일 생성 완료: {target_path}")
            else:
                print(f"⚠️ 템플릿 파일을 찾을 수 없습니다: {template_path}")
                
        except Exception as e:
            errors.append(f"slicer_opt.list 파일 생성 중 오류: {str(e)}")
            print(f"⚠️ slicer_opt.list 파일 생성 실패: {e}")
        
        # 데이터베이스 추가는 훈련 완료 시점으로 이동됨
        
        # 결과 반환
        result = {
            "message": f"{len(uploaded_files)}개 파일이 성공적으로 업로드되었습니다.",
            "exp_name": exp_name,
            "upload_dir": upload_dir,
            "uploaded_files": uploaded_files,
        }
        
        # slicer_opt.list 파일 생성 여부 확인
        slicer_path = f"{upload_dir}/slicer_opt.list"
        if os.path.exists(slicer_path):
            result["slicer_opt_created"] = True
            result["slicer_opt_path"] = slicer_path
        else:
            result["slicer_opt_created"] = False
        
        # 데이터베이스는 훈련 완료 후 자동 추가됨
        
        if errors:
            result["errors"] = errors
            result["message"] += f" {len(errors)}개 파일에서 오류가 발생했습니다."
        
        # 업로드가 성공했고 파일이 하나 이상 있으면 자동으로 훈련 시작
        if uploaded_files and len(uploaded_files) == 10:
            # 현재 훈련 중인지 확인
            global training_status
            if not training_status["is_training"]:
                # 훈련 요청 객체 생성
                train_request = TrainRequest(exp_name=exp_name)
                
                # 백그라운드에서 훈련 시작
                if background_tasks:
                    background_tasks.add_task(run_training_pipeline, train_request)
                    result["training_started"] = True
                    result["message"] += " 자동으로 훈련이 시작되었습니다."
                    print(f"🚀 업로드 완료 후 자동 훈련 시작: {exp_name}")
                else:
                    result["training_started"] = False
                    result["message"] += " 훈련 자동 시작을 위해서는 BackgroundTasks가 필요합니다."
            else:
                result["training_started"] = False
                result["message"] += " 이미 다른 훈련이 진행 중입니다."
        else:
            result["training_started"] = False
        
        return JSONResponse(
            status_code=200 if uploaded_files else 400,
            content=result
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "파일 업로드 중 오류가 발생했습니다.",
                "Exception": str(e)
            }
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
