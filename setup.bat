@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "CONF_FILE=%SCRIPT_DIR%.augmentum.conf"
set "ENV_FILE=%SCRIPT_DIR%.env"
set "ENV_COUNT=0"
set "HAS_ENV=0"
set "COMPOSE_PROFILES="
set "STT_WEBUI="
set "TTS_WEBUI="
set "EXISTING="

:: --- Terminal capabilities --------------------------------------------------
:: ANSI colors only under Windows Terminal (WT_SESSION set) where VT
:: processing is guaranteed on. Legacy conhost gets plain text instead of
:: escape-code garbage. The prompt trick below yields a literal ESC char.
set "ESC="
if defined WT_SESSION for /F "delims=#" %%E in ('"prompt #$E# & for %%E in (1) do rem"') do set "ESC=%%E"
set "C_BOLD=" & set "C_DIM=" & set "C_CYAN=" & set "C_GREEN=" & set "C_YELLOW=" & set "C_RESET="
if defined ESC (
  set "C_BOLD=%ESC%[1m"
  set "C_DIM=%ESC%[90m"
  set "C_CYAN=%ESC%[36m"
  set "C_GREEN=%ESC%[32m"
  set "C_YELLOW=%ESC%[33m"
  set "C_RESET=%ESC%[0m"
)

:: GPU sniff — used only to pick the default answer; the user always confirms.
set "GPU_NAME="
set "GPU_VRAM_MB="
where nvidia-smi >nul 2>&1 && (
  for /f "usebackq delims=" %%g in (`nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul`) do if not defined GPU_NAME set "GPU_NAME=%%g"
  for /f "usebackq delims=" %%v in (`nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader^,nounits 2^>nul`) do if not defined GPU_VRAM_MB set "GPU_VRAM_MB=%%v"
)
if not defined GPU_VRAM_MB set "GPU_VRAM_MB=0"
set /a GPU_VRAM_MB=GPU_VRAM_MB+0 2>nul

:: --- Header ------------------------------------------------------------------
echo.
echo       !C_CYAN!/\!C_RESET!
echo      !C_CYAN!/\/\!C_RESET!     !C_BOLD!Augmentum!C_RESET!
echo     !C_CYAN!/\/\/\!C_RESET!    !C_DIM!Personal AI on hardware you own!C_RESET!
echo.
echo   !C_DIM!----------------------------------------------!C_RESET!
echo   !C_DIM!8 steps - press Enter to accept the recommended default!C_RESET!
echo   !C_DIM!----------------------------------------------!C_RESET!

:: Check for existing config
if exist "%CONF_FILE%" (
  set /p EXISTING=<"%CONF_FILE%"
  echo.
  echo   !C_YELLOW!Existing configuration found:!C_RESET!
  echo     !C_DIM!!EXISTING!!C_RESET!
  echo.
  set /p RECONF="  > Reconfigure? [Y/n]: "
  if /i "!RECONF!"=="n" (
    echo.
    echo   Keeping existing configuration. Run start.bat to launch.
    echo.
    exit /b 0
  )
)

:: ============================================================
:: Step 1: Install type
:: ============================================================
echo.
echo.
echo   !C_DIM!1/8!C_RESET!  !C_BOLD!Install Type!C_RESET!
echo     !C_DIM!Use Augmentum, or work on it?!C_RESET!
echo.
echo     !C_CYAN!1!C_RESET!  Standard -- prebuilt image, no compile step (recommended)
echo     2  Contributor -- build from source, live-edit .\augmentum and .\ui
echo.

:install_type_choice
set "INSTALL_TYPE=1"
set /p INSTALL_TYPE="  > Install type [1-2 - Enter = 1]: "
if "%INSTALL_TYPE%"=="1" goto install_standard
if "%INSTALL_TYPE%"=="2" goto install_contributor
echo     Please enter 1 or 2
goto install_type_choice

:install_standard
set "DEV_OVERLAY="
set "SUM_INSTALL=Standard (prebuilt image)"
echo   !C_GREEN!*!C_RESET! Standard install
goto step_backend

:install_contributor
set "DEV_OVERLAY= compose.dev.yaml"
set "SUM_INSTALL=Contributor (build from source)"
echo   !C_GREEN!*!C_RESET! Contributor install -- first start builds the image (several minutes)
goto step_backend

:: ============================================================
:: Step 2: Backend selection
:: ============================================================
:step_backend
echo.
echo.
echo   !C_DIM!2/8!C_RESET!  !C_BOLD!Model Backend!C_RESET!
echo     !C_DIM!How will you run LLMs?!C_RESET!
echo.
echo     !C_CYAN!1!C_RESET!  Built-in engine -- bundled llama-server, models managed in
echo        the UI: speculative decoding, continuous batching, GGUF
echo        auto-discovery (recommended)
echo     2  External -- I already run Ollama, LM Studio, or another LLM
echo        server. Auto-detected, or configure in Settings later.
echo.

:backend_choice
set "BACKEND=1"
set /p BACKEND="  > Backend [1-2 - Enter = 1]: "
if "%BACKEND%"=="1" goto backend_engine
if "%BACKEND%"=="2" goto backend_skip
echo     Please enter 1 or 2
goto backend_choice

:backend_engine
echo   !C_GREEN!*!C_RESET! Built-in engine
echo.
if defined GPU_NAME echo     !C_DIM!Detected: !GPU_NAME!!C_RESET!
echo     !C_CYAN!1!C_RESET!  GPU (NVIDIA) -- local image gen + faster LLM
echo     2  CPU only     -- no GPU required
echo.

:: Default follows the GPU sniff: detected -> 1, absent -> 2.
set "HW_DEFAULT=2"
if defined GPU_NAME set "HW_DEFAULT=1"

:hw_choice
set "HW=%HW_DEFAULT%"
set /p HW="  > Hardware [1-2 - Enter = %HW_DEFAULT%]: "
if "%HW%"=="1" goto hw_gpu
if "%HW%"=="2" goto hw_cpu
echo     Please enter 1 or 2
goto hw_choice

:hw_gpu
set "COMPOSE_FILES=compose.yaml!DEV_OVERLAY! compose.gpu.yaml"
set "NEEDS_GPU=1"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_VARIANT=gpu"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_DOCKERFILE=Dockerfile.gpu"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_ENGINE_MANAGED=true"
set "SUM_BACKEND=Built-in engine - GPU"
if defined GPU_NAME set "SUM_BACKEND=Built-in engine - GPU (!GPU_NAME!)"
echo   !C_GREEN!*!C_RESET! GPU variant
goto engine_models

:hw_cpu
set "COMPOSE_FILES=compose.yaml!DEV_OVERLAY!"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_VARIANT=cpu"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_DOCKERFILE=Dockerfile"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_ENGINE_MANAGED=true"
set "SUM_BACKEND=Built-in engine - CPU"
echo   !C_GREEN!*!C_RESET! CPU variant
echo     !C_DIM!Image generation needs a GPU -- cloud providers work via Settings.!C_RESET!
goto engine_models

:engine_models
echo.
echo     !C_DIM!Augmentum auto-discovers GGUF models (LM Studio, HuggingFace cache).!C_RESET!
echo     !C_DIM!Optionally mount another folder of .gguf files now. Example: D:\Models!C_RESET!
set /p MODEL_DIR="  > Path to GGUF models (Enter to skip): "
if "!MODEL_DIR!"=="" goto engine_no_mount
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_ENGINE_MODEL_DIR=/data/host-models"
set "ENGINE_MODEL_MOUNT=!MODEL_DIR!"
set "SUM_MODELDIR=!MODEL_DIR!"
echo   !C_GREEN!*!C_RESET! Will mount !MODEL_DIR! into the container
goto step_images

:engine_no_mount
:: Skip means "no change", not "remove": a pre-existing model-dir config
:: (possibly pointing at a hand-edited mount override) survives the re-run.
set "OLD_ENGINE_MODEL_DIR="
if exist "%ENV_FILE%" (
  for /f "tokens=1,* delims==" %%a in ('findstr /b "AUGMENTUM_ENGINE_MODEL_DIR=" "%ENV_FILE%" 2^>nul') do set "OLD_ENGINE_MODEL_DIR=%%b"
)
if defined OLD_ENGINE_MODEL_DIR (
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_ENGINE_MODEL_DIR=!OLD_ENGINE_MODEL_DIR!"
  set "SUM_MODELDIR=!OLD_ENGINE_MODEL_DIR! [kept]"
  echo   !C_GREEN!*!C_RESET! Keeping existing model directory config
) else (
  echo     !C_DIM!Skipped. Add directories later in Settings ^> Manage Providers.!C_RESET!
)
goto step_images

:backend_skip
set "COMPOSE_FILES=compose.yaml!DEV_OVERLAY!"
set "HAS_ENV=0"
set "ENV_COUNT=0"
set "SUM_BACKEND=External (auto-detect)"
echo   !C_GREEN!*!C_RESET! External backend -- local servers are auto-detected on startup
goto step_images

:: ============================================================
:: Step 3: Image generation
:: ============================================================
:step_images
echo.
echo.
echo   !C_DIM!3/8!C_RESET!  !C_BOLD!Image Generation!C_RESET!
echo     !C_DIM!Local Stable Diffusion / FLUX. Needs an NVIDIA GPU with 4+ GB VRAM.!C_RESET!
echo.
set "SUM_IMG=off"
set /p IMAGES="  > Enable image generation? [y/N]: "
if /i "%IMAGES%"=="y" (
  if not defined NEEDS_GPU set "COMPOSE_FILES=!COMPOSE_FILES! compose.gpu.yaml"
  set "SUM_IMG=on (local SD/FLUX)"
  echo   !C_GREEN!*!C_RESET! Image generation enabled
  echo.
  set /p HF_TOKEN="  > HuggingFace token (optional, Enter to skip): "
  if not "!HF_TOKEN!"=="" (
    set /a ENV_COUNT+=1
    set "ENV_!ENV_COUNT!=AUGMENTUM_HUGGINGFACE_TOKEN=!HF_TOKEN!"
    set "HAS_ENV=1"
  )
) else (
  echo   !C_GREEN!*!C_RESET! Skipped -- re-run setup anytime to enable
)

:: ============================================================
:: Step 4: Speech-to-Text
:: ============================================================
:step_stt
echo.
echo.
echo   !C_DIM!4/8!C_RESET!  !C_BOLD!Speech-to-Text!C_RESET!
echo     !C_DIM!Moonshine (English, streaming) is built in -- nothing to add for English.!C_RESET!
echo.
echo     !C_CYAN!1!C_RESET!  English only -- built-in Moonshine, no extra container (recommended)
echo     2  Multilingual -- faster-whisper via Speaches (99 languages, ~2 GB)
echo.

:stt_choice
set "STT=1"
set /p STT="  > Voice input [1-2 - Enter = 1]: "
if "%STT%"=="1" goto stt_english
if "%STT%"=="2" goto stt_multi
echo     Please enter 1 or 2
goto stt_choice

:stt_english
set "SUM_STT=Moonshine (built-in)"
echo   !C_GREEN!*!C_RESET! Built-in Moonshine (English, streaming)
goto step_tts

:stt_multi
set "COMPOSE_FILES=!COMPOSE_FILES! compose.speaches.yaml"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_STT_PROVIDER_URL=http://speaches:8000"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_STT_DEFAULT_MODEL=Systran/faster-whisper-small"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_STT_MODEL=Systran/faster-whisper-small"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_VOICE_MOONSHINE_ENABLED=false"
set "STT_WEBUI=http://localhost:6200"
set "SUM_STT=faster-whisper (99 languages)"
echo   !C_GREEN!*!C_RESET! Multilingual STT added (faster-whisper-small, 99 languages)
goto step_tts

:: ============================================================
:: Step 5: Text-to-Speech
:: ============================================================
:step_tts
echo.
echo.
echo   !C_DIM!5/8!C_RESET!  !C_BOLD!Text-to-Speech!C_RESET!
echo     !C_DIM!Kokoro (CPU, 54 voices, mixing) is built in -- extras add cloning/emotion.!C_RESET!
echo.
echo     !C_CYAN!1!C_RESET!  Built-in only -- Kokoro, no extra container (recommended)
echo     2  Chatterbox -- multilingual cloning, GPU recommended (~4 GB VRAM)
echo     3  Chatterbox Turbo -- fast English cloning, GPU recommended (~2 GB VRAM)
echo     4  Fish Speech -- emotion + cloning, GPU required (~8 GB VRAM, gated)
echo     5  Qwen3 TTS -- emotion-aware speech, GPU required (~6 GB VRAM)
echo     6  Sesame CSM -- companion's relational voice, GPU (~3-4 GB VRAM, gated)
echo.

:tts_choice
set "TTS=1"
set /p TTS="  > Voice output [1-6 - Enter = 1]: "
if "%TTS%"=="1" goto tts_builtin
if "%TTS%"=="2" goto tts_chatterbox
if "%TTS%"=="3" goto tts_chatterbox_turbo
if "%TTS%"=="4" goto tts_fish
if "%TTS%"=="5" goto tts_qwen
if "%TTS%"=="6" goto tts_csm
echo     Please enter 1, 2, 3, 4, 5, or 6
goto tts_choice

:tts_builtin
set "SUM_TTS=Kokoro (built-in)"
echo   !C_GREEN!*!C_RESET! Built-in Kokoro (CPU, voice mixing)
goto step_game_stream

:tts_chatterbox
set "COMPOSE_FILES=!COMPOSE_FILES! compose.chatterbox.yaml"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_TTS_CHATTERBOX_URL=http://chatterbox:4123"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_CHATTERBOX_DEVICE=auto"
set "TTS_WEBUI=http://localhost:6400"
set "SUM_TTS=Chatterbox + Kokoro fallback"
echo   !C_GREEN!*!C_RESET! Chatterbox added -- Kokoro stays as fallback
echo     !C_DIM!500M multilingual model: 23+ languages, highest-quality cloning.!C_RESET!
echo     !C_DIM!Upload voice samples in Settings ^> Voice or the Chatterbox WebUI.!C_RESET!
goto step_game_stream

:tts_chatterbox_turbo
set "COMPOSE_FILES=!COMPOSE_FILES! compose.chatterbox-turbo.yaml"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_TTS_CHATTERBOX_TURBO_URL=http://chatterbox-turbo:8890"
set "SUM_TTS=Chatterbox Turbo + Kokoro fallback"
echo   !C_GREEN!*!C_RESET! Chatterbox Turbo added -- Kokoro stays as fallback
echo     !C_DIM!350M English model: lower VRAM, faster, [laugh]/[cough]/[chuckle] tags.!C_RESET!
echo     !C_DIM!Upload voice samples in Settings ^> Voice.!C_RESET!
goto step_game_stream

:tts_fish
set "COMPOSE_FILES=!COMPOSE_FILES! compose.fish-tts.yaml"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_TTS_FISH_URL=http://fish-tts:8080"
set "SUM_TTS=Fish Speech + Kokoro fallback"
echo   !C_GREEN!*!C_RESET! Fish Speech added -- Kokoro stays as fallback
echo     !C_YELLOW!Needs ~8 GB VRAM and a HuggingFace token (gated model).!C_RESET!
echo     !C_DIM!Accept the license: huggingface.co/fishaudio/openaudio-s1-mini!C_RESET!
echo     !C_DIM!Set the HF token in Settings ^> General after startup if not set above.!C_RESET!
echo     !C_DIM!48 inline emotion tags, cloning, 13+ languages. Enable!C_RESET!
echo     !C_DIM!"Emotion-aware TTS" in Settings ^> Voice for expressive speech.!C_RESET!
goto step_game_stream

:tts_qwen
set "COMPOSE_FILES=!COMPOSE_FILES! compose.qwen-tts.yaml"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_TTS_QWEN_URL=http://qwen-tts:8880"
echo     !C_DIM!Downloading Qwen3 TTS server...!C_RESET!
if exist "%SCRIPT_DIR%services\qwen-tts\.git" (
  git -C "%SCRIPT_DIR%services\qwen-tts" pull --quiet
) else (
  git clone --depth 1 https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git "%SCRIPT_DIR%services\qwen-tts"
)
set "SUM_TTS=Qwen3 TTS + Kokoro fallback"
echo   !C_GREEN!*!C_RESET! Qwen3 TTS added -- Kokoro stays as fallback
echo     !C_DIM!Tip: enable "Emotion-aware TTS" in Settings ^> Voice for expressive speech.!C_RESET!
goto step_game_stream

:tts_csm
set "COMPOSE_FILES=!COMPOSE_FILES! compose.sesame-csm.yaml"
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_TTS_SESAME_CSM_URL=http://sesame-csm:8920"
set "SUM_TTS=Sesame CSM (companion) + Kokoro"
echo   !C_GREEN!*!C_RESET! Sesame CSM added as the companion's voice
echo     !C_YELLOW!Needs a HuggingFace token + acceptance of TWO gated licenses:!C_RESET!
echo     !C_DIM!  huggingface.co/sesame/csm-1b!C_RESET!
echo     !C_DIM!  huggingface.co/meta-llama/Llama-3.2-1B!C_RESET!
echo     !C_DIM!Set the HF token in Settings ^> General after startup if not set above.!C_RESET!
echo     !C_DIM!This is a relational voice, not general TTS: prosody conditions on!C_RESET!
echo     !C_DIM!the conversation AND how you just sounded. Best for companion chat;!C_RESET!
echo     !C_DIM!narration and read-aloud stay on Kokoro. First boot pulls ~3-4 GB.!C_RESET!
echo     !C_DIM!Pick the voice in Settings ^> Companion; clone via Settings ^> Voice.!C_RESET!
goto step_game_stream

:: ============================================================
:: Step 6: Game Streaming (AGSP)
:: ============================================================
:step_game_stream
echo.
echo.
echo   !C_DIM!6/8!C_RESET!  !C_BOLD!Game Streaming!C_RESET!
echo     !C_DIM!Browser-streamed open-source games via WebRTC (Luanti today, more later).!C_RESET!
echo     !C_DIM!Adds ~500 MB of images; ~2 GB RAM + 2 CPUs per active session; NVENC best.!C_RESET!
echo.
set "SUM_GAME=off"
set /p GAMESTREAM="  > Enable game streaming? [y/N]: "
if /i "%GAMESTREAM%"=="y" (
  set "COMPOSE_FILES=!COMPOSE_FILES! compose.game-stream.yaml"
  set "HAS_ENV=1"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_GAME_STREAM_ENABLED=true"
  set "SUM_GAME=on"
  echo   !C_GREEN!*!C_RESET! Game streaming enabled -- build the images with: start.bat build
) else (
  echo   !C_GREEN!*!C_RESET! Skipped -- re-run setup anytime to enable
)

:: ============================================================
:: Step 7: HTTPS (always-on; informational only)
:: ============================================================
:step_https
echo.
echo.
echo   !C_DIM!7/8!C_RESET!  !C_BOLD!HTTPS!C_RESET!
echo     !C_DIM!On by default: Caddy serves HTTPS on 6443 with a self-signed cert that!C_RESET!
echo     !C_DIM!auto-regenerates when your LAN IP changes. HTTP stays on 6100 for!C_RESET!
echo     !C_DIM!loopback. Phones need HTTPS for mic/camera -- accept the cert once.!C_RESET!
echo   !C_GREEN!*!C_RESET! HTTPS on 6443 - HTTP on 6100

:: Caddy is no longer profile-gated; COMPOSE_PROFILES stays empty for
:: backward compatibility with existing .env files.
set "COMPOSE_PROFILES="

:: ============================================================
:: Step 8: Save config
:: ============================================================
echo.
echo.
echo   !C_DIM!8/8!C_RESET!  !C_BOLD!Finishing up!C_RESET!
echo.

:: Generate engine model mount override if user specified a model directory.
:: ----------------------------------------------------------------------
:: This file is the most-customized compose override in practice — users
:: with WSL ext4 drives, multi-drive setups, or specific permission needs
:: hand-edit it. The auto-generated form is a single-line, read-only
:: bind that won't satisfy those cases. So:
::   - If the file doesn't exist, generate the auto form.
::   - If it exists with the auto-gen header, regenerate (user hasn't
::     touched it, regenerating reflects the latest setup answers).
::   - If it exists WITHOUT the auto-gen header, it's hand-edited;
::     leave it alone and just add it to COMPOSE_FILES so the user's
::     work is honored on every subsequent re-run.
if defined ENGINE_MODEL_MOUNT (
  set "ENGINE_OVERRIDE=%SCRIPT_DIR%compose.engine-models.yaml"
  set "OVERRIDE_IS_AUTO=1"
  if exist "!ENGINE_OVERRIDE!" (
    :: Detect auto-generated form by header. Read first line only via a
    :: counter rather than goto-from-inside-for (which has flaky semantics
    :: across batch versions).
    set "OVERRIDE_IS_AUTO=0"
    set "_FIRST_LINE_DONE=0"
    for /f "usebackq delims=" %%h in ("!ENGINE_OVERRIDE!") do (
      if "!_FIRST_LINE_DONE!"=="0" (
        echo %%h | findstr /b /c:"# Generated by setup" >nul && set "OVERRIDE_IS_AUTO=1"
        set "_FIRST_LINE_DONE=1"
      )
    )
  )
  if "!OVERRIDE_IS_AUTO!"=="1" (
    echo # Generated by setup.bat — mounts host model directory> "!ENGINE_OVERRIDE!"
    echo services:>> "!ENGINE_OVERRIDE!"
    echo   augmentum:>> "!ENGINE_OVERRIDE!"
    echo     volumes:>> "!ENGINE_OVERRIDE!"
    echo       - !ENGINE_MODEL_MOUNT!:/data/host-models:ro>> "!ENGINE_OVERRIDE!"
    echo     !C_DIM!Generated compose.engine-models.yaml for the model mount.!C_RESET!
  ) else (
    echo     !C_DIM!Existing hand-edited compose.engine-models.yaml preserved.!C_RESET!
  )
  set "COMPOSE_FILES=!COMPOSE_FILES! compose.engine-models.yaml"
)

:: The override's existence implies intent, independent of this run's
:: answers — skipping the model-dir question must not drop hand-edited
:: mounts from the stack.
if exist "%SCRIPT_DIR%compose.engine-models.yaml" (
  echo !COMPOSE_FILES! | "%SystemRoot%\System32\findstr.exe" /c:"compose.engine-models.yaml" >nul || (
    set "COMPOSE_FILES=!COMPOSE_FILES! compose.engine-models.yaml"
    echo     !C_DIM!Including existing compose.engine-models.yaml [model mounts].!C_RESET!
  )
)

:: Carry over compose files the wizard doesn't manage (calling, rsshub,
:: hand-added overlays) from the previous configuration. The wizard only
:: rebuilds what it asked about.
if defined EXISTING (
  for %%f in (!EXISTING!) do (
    set "_KNOWN="
    for %%k in (compose.yaml compose.dev.yaml compose.gpu.yaml compose.speaches.yaml compose.chatterbox.yaml compose.chatterbox-turbo.yaml compose.fish-tts.yaml compose.qwen-tts.yaml compose.sesame-csm.yaml compose.game-stream.yaml compose.engine-models.yaml compose.classifier.yaml) do (
      if /i "%%f"=="%%k" set "_KNOWN=1"
    )
    if not defined _KNOWN (
      echo !COMPOSE_FILES! | "%SystemRoot%\System32\findstr.exe" /c:"%%f" >nul
      if errorlevel 1 (
        set "COMPOSE_FILES=!COMPOSE_FILES! %%f"
        echo     !C_DIM!Carried over %%f from previous configuration.!C_RESET!
      )
    )
  )
)

:: Classifier model — the voice/utility/vision workhorse. VRAM-aware default;
:: user confirms. Writes the external classifier container's -hf model (the
:: working auto-pull path) AND the managed "Slot C" defaults. Gemma E2B/E4B are
:: vision-capable (their mmproj makes the slot the captioner); SmolLM2 is
:: text-only (CPU tier).
set "CLF_DEFAULT=1"
if defined GPU_NAME if !GPU_VRAM_MB! GEQ 10000 set "CLF_DEFAULT=3"
if defined GPU_NAME if !GPU_VRAM_MB! GEQ 5000 if !GPU_VRAM_MB! LSS 10000 set "CLF_DEFAULT=2"
echo.
echo   Classifier model -- voice/utility/vision workhorse ^(!GPU_VRAM_MB! MB VRAM^)
echo     1^) SmolLM2-135M -- CPU, text-only ^(no vision^)
echo     2^) Gemma-4-E2B -- GPU, fast + can see ^(~5 GB^)
echo     3^) Gemma-4-E4B -- GPU, best + vision ^(~10 GB^)
set /p CLF="  > Classifier model [1-3 - Enter = !CLF_DEFAULT!]: "
if not defined CLF set "CLF=!CLF_DEFAULT!"
set "CLF_GPU=0"
if "!CLF!"=="2" (
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_HF=unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_NGL=99"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_CTX=32768"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE=1.0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P=0.95"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K=64"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_VISION_ARGS=--mmproj-url https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/mmproj-F16.gguf"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SLOT_MODEL=gemma-4-E2B-it-qat-UD-Q4_K_XL"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS=99"
  set "CLF_GPU=1"
  set "SUM_CLF=Gemma-4-E2B (GPU, vision)"
) else if "!CLF!"=="3" (
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_HF=unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_NGL=99"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_CTX=32768"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE=1.0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P=0.95"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K=64"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_VISION_ARGS=--mmproj-url https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/mmproj-F16.gguf"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SLOT_MODEL=gemma-4-E4B-it-qat-UD-Q4_K_XL"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS=99"
  set "CLF_GPU=1"
  set "SUM_CLF=Gemma-4-E4B (GPU, vision)"
) else (
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_HF=bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_NGL=0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE=0.0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P=1.0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K=0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SLOT_MODEL=SmolLM2-135M-Instruct-Q8_0"
  set /a ENV_COUNT+=1
  set "ENV_!ENV_COUNT!=AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS=0"
  set "SUM_CLF=SmolLM2-135M (CPU, text-only)"
)

:: Local voice/intent classifier — built-in companion resource, default-on.
:: Idempotent append so every fresh/reconfigured install gets the container.
echo !COMPOSE_FILES! | "%SystemRoot%\System32\findstr.exe" /c:"compose.classifier.yaml" >nul
if errorlevel 1 set "COMPOSE_FILES=!COMPOSE_FILES! compose.classifier.yaml"
:: GPU classifier -> add the CUDA overlay so the container actually offloads.
if "!CLF_GPU!"=="1" (
  echo !COMPOSE_FILES! | "%SystemRoot%\System32\findstr.exe" /c:"compose.classifier-gpu.yaml" >nul
  if errorlevel 1 set "COMPOSE_FILES=!COMPOSE_FILES! compose.classifier-gpu.yaml"
)

:: Save config
echo !COMPOSE_FILES!> "%CONF_FILE%"

:: SearXNG session secret — preserve existing value across reconfigures,
:: otherwise generate a 256-bit hex value via PowerShell's crypto RNG.
:: PowerShell ships with Windows 10+ so this works on every supported box.
set "SEARXNG_SECRET="
if exist "%ENV_FILE%" (
  for /f "tokens=1,* delims==" %%a in ('findstr /b "SEARXNG_SECRET=" "%ENV_FILE%" 2^>nul') do (
    set "SEARXNG_SECRET=%%b"
  )
)
:: Treat the literal placeholder as unset — it shouldn't survive setup.
if /i "!SEARXNG_SECRET!"=="change-me-in-production" set "SEARXNG_SECRET="
if not defined SEARXNG_SECRET (
  for /f %%a in ('powershell -NoProfile -Command "$rng=New-Object System.Security.Cryptography.RNGCryptoServiceProvider; $b=New-Object byte[] 32; $rng.GetBytes($b); [BitConverter]::ToString($b).Replace('-','').ToLower()"') do set "SEARXNG_SECRET=%%a"
)
set "HAS_ENV=1"
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=SEARXNG_SECRET=!SEARXNG_SECRET!"

:: TURN shared secret (coturn --use-auth-secret HMAC). See setup.sh for
:: full background; the proxy mints ephemeral creds against this, both
:: sides must agree. Documented dev default in compose.calling.yaml is
:: a placeholder — any deployment beyond localhost must rotate it.
set "AUGMENTUM_TURN_SECRET="
if exist "%ENV_FILE%" (
  for /f "tokens=1,* delims==" %%a in ('findstr /b "AUGMENTUM_TURN_SECRET=" "%ENV_FILE%" 2^>nul') do (
    set "AUGMENTUM_TURN_SECRET=%%b"
  )
)
if /i "!AUGMENTUM_TURN_SECRET!"=="augmentum-turn-dev-secret-change-in-env" set "AUGMENTUM_TURN_SECRET="
if not defined AUGMENTUM_TURN_SECRET (
  for /f %%a in ('powershell -NoProfile -Command "$rng=New-Object System.Security.Cryptography.RNGCryptoServiceProvider; $b=New-Object byte[] 32; $rng.GetBytes($b); [BitConverter]::ToString($b).Replace('-','').ToLower()"') do set "AUGMENTUM_TURN_SECRET=%%a"
)
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=AUGMENTUM_TURN_SECRET=!AUGMENTUM_TURN_SECRET!"

:: LiveKit API key — opaque short identifier (not a secret). Use a clear
:: prefix so it's distinguishable from the secret in logs / configs.
set "LIVEKIT_API_KEY="
if exist "%ENV_FILE%" (
  for /f "tokens=1,* delims==" %%a in ('findstr /b "LIVEKIT_API_KEY=" "%ENV_FILE%" 2^>nul') do (
    set "LIVEKIT_API_KEY=%%b"
  )
)
if /i "!LIVEKIT_API_KEY!"=="devkey" set "LIVEKIT_API_KEY="
if not defined LIVEKIT_API_KEY (
  for /f %%a in ('powershell -NoProfile -Command "$rng=New-Object System.Security.Cryptography.RNGCryptoServiceProvider; $b=New-Object byte[] 8; $rng.GetBytes($b); 'aug_' + [BitConverter]::ToString($b).Replace('-','').ToLower()"') do set "LIVEKIT_API_KEY=%%a"
)
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=LIVEKIT_API_KEY=!LIVEKIT_API_KEY!"

:: LiveKit API secret — the actual shared secret for JWT signing /
:: validation. Proxy + livekit container both read this from .env.
set "LIVEKIT_API_SECRET="
if exist "%ENV_FILE%" (
  for /f "tokens=1,* delims==" %%a in ('findstr /b "LIVEKIT_API_SECRET=" "%ENV_FILE%" 2^>nul') do (
    set "LIVEKIT_API_SECRET=%%b"
  )
)
if /i "!LIVEKIT_API_SECRET!"=="augmentum-livekit-dev-secret-change-in-env" set "LIVEKIT_API_SECRET="
if not defined LIVEKIT_API_SECRET (
  for /f %%a in ('powershell -NoProfile -Command "$rng=New-Object System.Security.Cryptography.RNGCryptoServiceProvider; $b=New-Object byte[] 32; $rng.GetBytes($b); [BitConverter]::ToString($b).Replace('-','').ToLower()"') do set "LIVEKIT_API_SECRET=%%a"
)
set /a ENV_COUNT+=1
set "ENV_!ENV_COUNT!=LIVEKIT_API_SECRET=!LIVEKIT_API_SECRET!"

:: Build COMPOSE_FILE from compose file list (semicolon-separated for Windows)
set "COMPOSE_FILE_VAR="
for %%f in (!COMPOSE_FILES!) do (
  if "!COMPOSE_FILE_VAR!"=="" (
    set "COMPOSE_FILE_VAR=%%f"
  ) else (
    set "COMPOSE_FILE_VAR=!COMPOSE_FILE_VAR!;%%f"
  )
)

:: Preserve user customizations across re-runs.
:: ----------------------------------------------------------------------
:: Setup actively manages a fixed set of keys (see managed-set below). Any
:: OTHER key=value lines in the existing .env are user-set — typical
:: examples: AUGMENTUM_BIND_HOST, AUGMENTUM_LLAMACPP_MODEL_DIR,
:: AUGMENTUM_ENGINE_EXTRA_MODEL_DIRS, AUGMENTUM_DEFAULT_BACKEND, profile
:: API keys, etc. Without preservation, every re-run silently wipes them
:: and the user has to re-do customizations from memory. That's a real
:: footgun for an OSS project.
::
:: We use PowerShell rather than batch because the managed-set check
:: needs proper string ops; also keeps a single backup at .env.bak so
:: the user always has a safety net.
set "USER_OVERRIDES=%TEMP%\augmentum_user_env.txt"
del /q "%USER_OVERRIDES%" 2>nul
if exist "%ENV_FILE%" (
  copy /y "%ENV_FILE%" "%ENV_FILE%.bak" >nul 2>&1
  powershell -NoProfile -Command ^
    "$managed=@('COMPOSE_FILE','COMPOSE_PROFILES','SEARXNG_SECRET','AUGMENTUM_TURN_SECRET','LIVEKIT_API_KEY','LIVEKIT_API_SECRET','AUGMENTUM_VARIANT','AUGMENTUM_DOCKERFILE','AUGMENTUM_ENGINE_MANAGED','AUGMENTUM_ENGINE_MODEL_DIR','AUGMENTUM_HUGGINGFACE_TOKEN','AUGMENTUM_STT_PROVIDER_URL','AUGMENTUM_STT_DEFAULT_MODEL','AUGMENTUM_STT_MODEL','AUGMENTUM_VOICE_MOONSHINE_ENABLED','AUGMENTUM_TTS_CHATTERBOX_URL','AUGMENTUM_CHATTERBOX_DEVICE','AUGMENTUM_TTS_CHATTERBOX_TURBO_URL','AUGMENTUM_TTS_FISH_URL','AUGMENTUM_TTS_QWEN_URL','AUGMENTUM_TTS_SESAME_CSM_URL','AUGMENTUM_GAME_STREAM_ENABLED','AUGMENTUM_CLASSIFIER_HF','AUGMENTUM_CLASSIFIER_NGL','AUGMENTUM_CLASSIFIER_CTX','AUGMENTUM_CLASSIFIER_SAMPLING_TEMPERATURE','AUGMENTUM_CLASSIFIER_SAMPLING_TOP_P','AUGMENTUM_CLASSIFIER_SAMPLING_TOP_K','AUGMENTUM_CLASSIFIER_VISION_ARGS','AUGMENTUM_CLASSIFIER_SLOT_MODEL','AUGMENTUM_CLASSIFIER_SLOT_GPU_LAYERS');" ^
    "$pattern='^(' + ($managed -join '|') + ')=';" ^
    "Get-Content '%ENV_FILE%' | Where-Object { ($_ -match '^[A-Z][A-Z0-9_]*=' -and $_ -notmatch $pattern) -or ($_ -notmatch '^[A-Z][A-Z0-9_]*=' -and $_ -notmatch '^# Generated by Augmentum setup' -and $_ -notmatch '^# User customizations preserved from previous') } | Set-Content -Encoding ASCII '%USER_OVERRIDES%'" <nul
)

echo # Generated by Augmentum setup> "%ENV_FILE%"
echo COMPOSE_FILE=!COMPOSE_FILE_VAR!>> "%ENV_FILE%"
for /L %%i in (1,1,!ENV_COUNT!) do (
  echo !ENV_%%i!>> "%ENV_FILE%"
)

:: Append preserved user customizations after the managed block.
if exist "%USER_OVERRIDES%" (
  rem Full path: a Git Bash / MSYS-spawned cmd resolves bare `find` to GNU
  rem find (which would crawl the filesystem); System32's is the line counter.
  for /f %%s in ('type "%USER_OVERRIDES%" ^| "%SystemRoot%\System32\find.exe" /c /v ""') do set "OVERRIDE_COUNT=%%s"
  if not "!OVERRIDE_COUNT!"=="0" (
    echo.>> "%ENV_FILE%"
    echo # User customizations preserved from previous .env>> "%ENV_FILE%"
    type "%USER_OVERRIDES%">> "%ENV_FILE%"
    echo     !C_DIM!Preserved !OVERRIDE_COUNT! user customization^(s^); previous .env backed up to .env.bak!C_RESET!
  )
  del /q "%USER_OVERRIDES%" 2>nul
)

:: --- Summary panel ----------------------------------------------------------
echo.
echo   !C_DIM!--------------------------------------------------!C_RESET!
echo   !C_GREEN!* Setup complete!C_RESET!
echo.
echo     !C_DIM!Install         !C_RESET!!SUM_INSTALL!
echo     !C_DIM!Backend         !C_RESET!!SUM_BACKEND!
if defined SUM_CLF echo     !C_DIM!Classifier      !C_RESET!!SUM_CLF!
if defined SUM_MODELDIR echo     !C_DIM!Model dir       !C_RESET!!SUM_MODELDIR!
echo     !C_DIM!Image gen       !C_RESET!!SUM_IMG!
echo     !C_DIM!Speech-to-text  !C_RESET!!SUM_STT!
echo     !C_DIM!Text-to-speech  !C_RESET!!SUM_TTS!
echo     !C_DIM!Game streaming  !C_RESET!!SUM_GAME!
echo     !C_DIM!HTTPS           !C_RESET!on - 6443 (self-signed)
echo     !C_DIM!Saved           !C_RESET!.augmentum.conf - .env
echo   !C_DIM!--------------------------------------------------!C_RESET!
echo.
echo   !C_BOLD!Open!C_RESET!            !C_CYAN!https://localhost:6443!C_RESET!
echo   !C_BOLD!From your LAN!C_RESET!   !C_CYAN!https://^<your-lan-ip^>:6443!C_RESET!  !C_DIM!(accept cert once per device)!C_RESET!
if not "!STT_WEBUI!"=="" (
  echo   !C_BOLD!STT WebUI!C_RESET!       !C_CYAN!!STT_WEBUI!!C_RESET!
)
if not "!TTS_WEBUI!"=="" (
  echo   !C_BOLD!TTS WebUI!C_RESET!       !C_CYAN!!TTS_WEBUI!!C_RESET!
)
echo.
echo   !C_DIM!Commands: start.bat (start) - start.bat -d (background) - start.bat down (stop)!C_RESET!
echo   !C_DIM!          start.bat logs (logs) - setup.bat (reconfigure)!C_RESET!
echo.

set /p LAUNCH="  > Start Augmentum now? [Y/n]: "
if /i "%LAUNCH%"=="n" (
  echo.
  echo   Run start.bat when you're ready.
  echo.
  exit /b 0
)

echo.
echo   Starting...
echo.
call "%SCRIPT_DIR%start.bat"
