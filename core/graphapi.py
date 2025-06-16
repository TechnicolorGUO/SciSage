from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import asyncio
from typing import Any, Dict

from langgraph_service import LangGraphService
from fastapi.middleware.cors import CORSMiddleware

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义 FastAPI 应用
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # 允许的源列表
    allow_origins=["http://localhost:63342"],  # 或者 ["*"] 允许所有源
    allow_credentials=True,    # 允许携带cookie
    allow_methods=["*"],      # 允许的HTTP方法，或者具体指定 ["POST", "OPTIONS"]
    allow_headers=["*"],      # 允许的HTTP头，或者具体指定 ["Content-Type"]
)

# 定义请求和响应模型
class ReportRequest(BaseModel):
    topic: str
    request_id: str


class ReportResponse(BaseModel):
    #request_id: str
    final_report: str
    #generated_at: datetime


# 异步生成报告
async def process_report(topic: str, request_id: str) -> Dict[str, Any]:
    """
    Process report generation using LangGraph service.
    """
    try:
        # 准备输入状态
        input_state = {"topic": topic}
        graph_service = LangGraphService()

        # 直接使用 ainvoke
        result = await graph_service.graph.ainvoke(input_state)

        if not result or "final_report" not in result:
            raise ValueError("No report generated")

        return {
            "request_id": request_id,
            "final_report": result["final_report"],
            "generated_at": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error processing report for request {request_id}: {str(e)}")
        raise


# 定义接口
@app.post("/generate-report_async", response_model=ReportResponse)
async def generate_report_endpoint(request: ReportRequest):
    """
    接收请求并返回生成的报告。
    """
    try:
        # 调用处理函数生成报告
        report = await process_report(request.topic, request.request_id)

        # 将字典数据转换为 ReportResponse 模型
        response = ReportResponse(
            request_id=report["request_id"],
            final_report=report["final_report"],
            generated_at=report["generated_at"]
        )
        print("接口应答：")
        print(response)
        return response
    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        logger.error(f"Unexpected error in endpoint: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)

@app.post("/generate-report", response_model=ReportResponse)
def generate_report_endpoint(request: ReportRequest):
    """
    FastAPI 接口，同步调用 LangGraphService 的 run_sync。
    """
    service = LangGraphService()
    logger.info(request)
    try:
        # 使用同步方法生成报告
        final_report = service.run_sync(request.topic)
        print("最终生成的报告")
        print(final_report)
        return ReportResponse(final_report=str(final_report))
    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        # 日志记录错误信息
        logger.error(f"Unexpected error in endpoint: {e}")
        return JSONResponse(content={"error": "Internal server error"}, status_code=500)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9099, timeout_keep_alive=600)


