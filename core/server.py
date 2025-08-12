# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

"""
Paper Generation Service - Async HTTP service for paper generation pipeline
"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import asyncio
import concurrent.futures
from threading import Thread

from main_workflow_opt_for_paper import PaperGenerationPipeline, save_results
from log import logger

# Pydantic models for request/response
class PaperGenerationRequest(BaseModel):
    user_name: str = Field(..., description="User name")
    user_query: str = Field(..., description="User query for paper generation")
    task_id: Optional[str] = Field(None, description="Optional task ID")
    output_dir: str = Field("temp", description="Output directory")
    rag_service_url: str = Field("http://localhost:8000", description="RAG service URL")
    # Pipeline configuration parameters
    outline_max_reflections: int = Field(2, description="Maximum outline reflections")
    outline_max_sections: int = Field(5, description="Maximum sections in outline")
    outline_min_depth: int = Field(1, description="Minimum outline depth")
    section_writer_model: str = Field("Qwen3-32B", description="Model for section writing")
    do_section_reflection: bool = Field(True, description="Enable section reflection")
    section_reflection_max_turns: int = Field(2, description="Max section reflection turns")
    do_global_reflection: bool = Field(True, description="Enable global reflection")
    global_reflection_max_turns: int = Field(2, description="Max global reflection turns")
    global_abstract_conclusion_max_turns: int = Field(2, description="Max abstract/conclusion turns")
    only_outline: bool = Field(False, description="Generate only outline")
    do_query_understand: bool = Field(True, description="Enable query understanding")
    return_only_final_paper: bool = Field(default=True, description="Return only final_paper content")
    translate_to_chinese: bool = Field(False, description="Translate to Chinses")



class PaperGenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str
    paper_title: Optional[str] = None
    execution_time: Optional[float] = None
    output_path: Optional[str] = None
    error: Optional[str] = None

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    current_state: str
    paper_title: Optional[str] = None
    error: Optional[str] = None
    execution_times: Dict[str, float] = {}

# Global task storage (in production, use Redis or database)
active_tasks: Dict[str, Dict[str, Any]] = {}

# FastAPI app initialization
app = FastAPI(
    title="Paper Generation Service",
    description="Async service for academic paper generation",
    version="1.0.0"
)

async def run_paper_generation(task_id: str, request: PaperGenerationRequest):
    """Background task to run paper generation"""
    try:
        logger.info(f"Starting paper generation task {task_id}")
        active_tasks[task_id]["status"] = "running"
        active_tasks[task_id]["start_time"] = time.time()

        # Convert request to kwargs
        kwargs = request.dict()
        user_name = kwargs.pop("user_name")
        user_query = kwargs.pop("user_query")
        output_dir = kwargs.pop("output_dir")
        rag_service_url = kwargs.pop("rag_service_url")
        return_only_final_paper = kwargs.pop("return_only_final_paper", True)
        kwargs.pop("task_id", None)

        # Create pipeline
        pipeline = PaperGenerationPipeline(
            user_name=user_name,
            user_query=user_query,
            task_id=task_id,
            output_dir=output_dir,
            rag_service_url=rag_service_url,
            **kwargs
        )

        # Generate paper
        results = await pipeline.generate_paper()

        # Calculate execution time
        execution_time = time.time() - active_tasks[task_id]["start_time"]

        # Process results based on return_only_final_paper option
        if return_only_final_paper and "final_paper" in results:
            final_paper_content = results["final_paper"].get("markdown_content", "")
            final_paper_content_zh = results["final_paper"].get("markdown_content_zh","")
            processed_results = {
                "final_paper_content": final_paper_content,
                "final_paper_content_zh": final_paper_content_zh,
                "paper_title": results.get("paper_title", ""),
                "task_id": task_id
            }
        else:
            processed_results = results

        # Save results
        output_path = save_results(results, output_dir)

        # Update task status
        if "error" in results:
            active_tasks[task_id].update({
                "status": "failed",
                "error": results["error"],
                "execution_time": execution_time,
                "results": processed_results
            })
            logger.error(f"Task {task_id} failed: {results['error']}")
        else:
            active_tasks[task_id].update({
                "status": "completed",
                "paper_title": results.get("paper_title", ""),
                "execution_time": execution_time,
                "output_path": output_path,
                "results": processed_results
            })
            logger.info(f"Task {task_id} completed successfully")

    except Exception as e:
        error_msg = f"Unexpected error in task {task_id}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        active_tasks[task_id].update({
            "status": "failed",
            "error": error_msg,
            "execution_time": time.time() - active_tasks[task_id].get("start_time", time.time())
        })



@app.post("/generate", response_model=PaperGenerationResponse)
async def generate_paper(request: PaperGenerationRequest, background_tasks: BackgroundTasks):
    """Start paper generation process"""
    try:
        # Generate task ID
        task_id = request.task_id or f"{request.user_name}_{str(uuid.uuid4())[:8]}"

        # Check if task already exists
        if task_id in active_tasks:
            raise HTTPException(
                status_code=400,
                detail=f"Task {task_id} already exists"
            )

        # Initialize task
        active_tasks[task_id] = {
            "task_id": task_id,
            "user_name": request.user_name,
            "user_query": request.user_query,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "request": request.dict()
        }

        # Start background task
        background_tasks.add_task(run_paper_generation, task_id, request)

        logger.info(f"Started paper generation task {task_id} for query: {request.user_query}")

        return PaperGenerationResponse(
            task_id=task_id,
            status="pending",
            message=f"Paper generation task {task_id} started successfully"
        )


    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting paper generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start task: {str(e)}")

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = active_tasks[task_id]

    # Get detailed progress if available
    execution_times = {}
    current_state = "unknown"

    if "results" in task:
        process_data = task["results"].get("process_data", {})
        execution_times = process_data.get("execution_times", {})
        current_state = process_data.get("process_state", "unknown")

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        current_state=current_state,
        paper_title=task.get("paper_title"),
        error=task.get("error"),
        execution_times=execution_times
    )

# 添加同步执行器
sync_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

def run_paper_generation_sync(task_id: str, request: PaperGenerationRequest):
    """同步运行论文生成"""
    try:
        # 在新的事件循环中运行异步代码
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 运行异步任务
            return loop.run_until_complete(_run_paper_generation_internal(task_id, request))
        finally:
            loop.close()

    except Exception as e:
        error_msg = f"Sync execution error in task {task_id}: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return {"error": error_msg}

async def _run_paper_generation_internal(task_id: str, request: PaperGenerationRequest):
    """内部异步执行函数"""
    logger.info(f"Starting paper generation task {task_id}")

    # Convert request to kwargs
    kwargs = request.dict()
    user_name = kwargs.pop("user_name")
    user_query = kwargs.pop("user_query")
    output_dir = kwargs.pop("output_dir")
    rag_service_url = kwargs.pop("rag_service_url")
    kwargs.pop("task_id", None)  # 移除 task_id 避免重复传递


    # Create pipeline
    pipeline = PaperGenerationPipeline(
        user_name=user_name,
        user_query=user_query,
        task_id=task_id,
        output_dir=output_dir,
        rag_service_url=rag_service_url,
        **kwargs
    )

    # Generate paper
    results = await pipeline.generate_paper()

    # Save results
    output_path = save_results(results, output_dir)

    return {
        "results": results,
        "output_path": output_path,
        "task_id": task_id
    }

@app.post("/generate_sync")
async def generate_paper_sync(request: PaperGenerationRequest):
    """同步论文生成接口（在线程池中执行）"""
    try:
        # Generate task ID
        task_id = request.task_id or f"{request.user_name}_{str(uuid.uuid4())[:8]}"

        logger.info(f"Starting sync paper generation task {task_id}")

        # 在线程池中执行同步任务
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            sync_executor,
            run_paper_generation_sync,
            task_id,
            request
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Process results based on return_only_final_paper option
        if request.return_only_final_paper and "final_paper" in result["results"]:
            final_paper_content = result["results"]["final_paper"].get("markdown_content", "")
            final_paper_content_zh = result["results"]["final_paper"].get("markdown_content_zh","")
            return {
                "task_id": task_id,
                "status": "completed",
                "paper_title": result["results"].get("paper_title", ""),
                # "output_path": result["output_path"],
                "final_paper_content": final_paper_content,
                "final_paper_content_zh": final_paper_content_zh
            }
        else:
            return {
                "task_id": task_id,
                "status": "completed",
                "paper_title": result["results"].get("paper_title", ""),
                # "output_path": result["output_path"],
                "results": result["results"]
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sync paper generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate paper: {str(e)}")


@app.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """Get task result"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = active_tasks[task_id]

    if task["status"] == "pending" or task["status"] == "running":
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task_id,
                "status": task["status"],
                "message": "Task is still running"
            }
        )

    if task["status"] == "failed":
        return JSONResponse(
            status_code=500,
            content={
                "task_id": task_id,
                "status": "failed",
                "error": task.get("error", "Unknown error")
            }
        )

    # Return full results
    return {
        "task_id": task_id,
        "status": task["status"],
        "execution_time": task.get("execution_time"),
        "output_path": task.get("output_path"),
        "results": task.get("results", {})
    }

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete task from memory"""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task_status = active_tasks[task_id]["status"]
    if task_status == "running":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete running task {task_id}"
        )

    del active_tasks[task_id]
    return {"message": f"Task {task_id} deleted successfully"}

@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    tasks_summary = []
    for task_id, task in active_tasks.items():
        tasks_summary.append({
            "task_id": task_id,
            "user_name": task.get("user_name"),
            "user_query": task.get("user_query", "")[:100] + "..." if len(task.get("user_query", "")) > 100 else task.get("user_query", ""),
            "status": task["status"],
            "created_at": task.get("created_at"),
            "execution_time": task.get("execution_time")
        })

    return {"tasks": tasks_summary, "total": len(tasks_summary)}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Paper Generation Service",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(active_tasks)
    }

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Paper Generation Service",
        "version": "1.0.0",
        "description": "Async service for academic paper generation",
        "endpoints": {
            "POST /generate": "Start paper generation",
            "GET /status/{task_id}": "Get task status",
            "GET /result/{task_id}": "Get task result",
            "GET /tasks": "List all tasks",
            "DELETE /task/{task_id}": "Delete task",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper Generation Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--log-level", default="info", help="Log level")

    args = parser.parse_args()

    logger.info(f"Starting Paper Generation Service on {args.host}:{args.port}")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
        reload=False
    )

    # python server.py --host 0.0.0.0 --port 8080
    # uvicorn server:app --host 0.0.0.0 --port 8080 --workers 1

