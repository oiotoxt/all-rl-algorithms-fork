# oox

```txt
--index-url https://download.pytorch.org/whl/cpu
numpy
matplotlib
torch
gymnasium
```

```bash
uv venv # uv 는 의도적으로 기본 pip를 설치하지 않음

va # .venv/bin/activate

# 먼저 requirements.txt 수정

uv pip install -r requirements.txt

# Test
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
#2.6.0+cpu
#None

# ipykernel 을 requirements.txt 써서 설치하면 이유는 모를 에러 발생. 그냥 별도로 설치
uv pip install ipykernel
```
