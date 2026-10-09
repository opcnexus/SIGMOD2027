# RepSpace artifact — 常用入口
PY ?= python3

.PHONY: help verify tables test all clean

help:
	@echo "make verify   # 工件完整性/合规性校验 + 单元测试（无需数据）"
	@echo "make tables   # 重建论文表格并与已发表数值比对（无需数据）"
	@echo "make test     # 仅跑链级指标与失败模式的回归测试"
	@echo "make all      # verify + tables"

verify:
	$(PY) scripts/verify_artifact.py

tables:
	$(PY) scripts/make_paper_tables.py --check

test:
	$(PY) tests/test_p0_fixes.py

all: verify tables

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
