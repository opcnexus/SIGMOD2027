#!/usr/bin/env python3
"""repspace API service (FastAPI)

Endpoints:
  GET /health                     health check
  GET /files                      list of queryable log files
  GET /messages?file=X&from=..&to=..  view messages
  POST /trace                     backtrack: given file+mid (or text), return parent chain + thread context
  POST /batch_trace               batch backtracking
Launch:
  python3 api_server.py            (default 0.0.0.0:8300)
"""
from __future__ import annotations

import os
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtrack_engine import BacktrackEngine, build_default_engine

app = FastAPI(title="Conversation Backtracking API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engine: BacktrackEngine = None  # lazily initialized


@app.on_event("startup")
async def startup():
    global engine
    engine = build_default_engine()


class TraceReq(BaseModel):
    file: Optional[str] = Field(None, description="log file name, e.g. 2016-02-22_17")
    mid: Optional[int] = Field(None, description="message id (0-based line number)")
    text: Optional[str] = Field(None, description="locate by text (mutually exclusive with file/mid)")
    max_hops: int = 10


class BatchReq(BaseModel):
    items: List[TraceReq]


@app.get("/health")
def health():
    return {"status": "ok", "files": len(engine.logs) if engine else 0}


@app.get("/files")
def files():
    return {"files": sorted(engine.by_name.keys())}


@app.get("/messages")
def messages(file: str, start: int = 1000, end: int = 1050):
    if file not in engine.by_name:
        raise HTTPException(404, "file not found: {}".format(file))
    log = engine.by_name[file]
    out = []
    for m in log.messages[start:end]:
        out.append({"mid": m.mid, "speaker": m.speaker, "text": m.text, "system": m.is_system})
    return {"file": file, "messages": out}


@app.post("/trace")
def trace(req: TraceReq):
    if req.text:
        hits = engine.locate(req.text)
        if not hits:
            raise HTTPException(404, "no matching message")
        fname, mid, s = hits[0]
        res = engine.trace(fname, mid, max_hops=req.max_hops)
    else:
        if not req.file or req.mid is None:
            raise HTTPException(400, "need file+mid or text")
        if req.file not in engine.by_name:
            raise HTTPException(404, "file not found")
        res = engine.trace(req.file, req.mid, max_hops=req.max_hops)
    import dataclasses
    return dataclasses.asdict(res)


@app.post("/batch_trace")
def batch_trace(req: BatchReq):
    import dataclasses
    out = []
    for item in req.items[:20]:
        try:
            if item.text:
                hits = engine.locate(item.text)
                if not hits:
                    out.append({"error": "no match"}); continue
                fname, mid, _ = hits[0]
            else:
                fname, mid = item.file, item.mid
            res = engine.trace(fname, mid, max_hops=item.max_hops)
            out.append(dataclasses.asdict(res))
        except Exception as e:
            out.append({"error": str(e)})
    return {"results": out}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8300))
    uvicorn.run(app, host="127.0.0.1", port=port)
