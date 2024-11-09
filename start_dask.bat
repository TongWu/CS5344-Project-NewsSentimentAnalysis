@echo off
REM 启动 Dask Scheduler
start cmd /k "dask-scheduler"

REM 等待 Scheduler 启动，确保 Workers 能够连接
timeout /t 5

REM 启动 8 个 Dask Workers，每个有 4 个线程和 1.5GB 内存限制
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"
start cmd /k "dask-worker tcp://127.0.0.1:8786 --nthreads 4 --memory-limit 1.5GB"

echo Dask Scheduler and Workers started.
pause
