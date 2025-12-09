##### MSD ######
uv run python -u scripts/benchmark_logm.py \
  --dataset MSD \
  --model Nystrom \
  --sigma 7 \
  --lam 1-8 \
  --m_stop 90000 


##### Mini Higgs ####
uv run python -u scripts/benchmark_logm.py \
  --dataset mini_Higgs \
  --model Nystrom \
  --sigma 3.8 \
  --lam 3e-8 \
  --m_stop 90000 

#### Higgs ####
uv run python -u scripts/benchmark_logm.py \
  --dataset HIGGS \
  --model Nystrom \
  --sigma 3.8 \
  --lam 3e-8 \
  --m_stop 90000 
