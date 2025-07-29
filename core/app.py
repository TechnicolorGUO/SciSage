#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : AI Assistant
# [Descriptions] : FastAPI server for paper generation pipeline
# ==================================================================

import asyncio
import json
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 添加这个导入
from pydantic import BaseModel, Field
from threading import Lock

from main_workflow_opt_for_paper import PaperGenerationPipeline, save_results
from log import logger


# Initialize FastAPI app
app = FastAPI(
    title="Academic Paper Generation API",
    description="API for generating academic papers using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
static_dir = "static"
os.makedirs(static_dir, exist_ok=True)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Global task storage
active_tasks: Dict[str, Dict[str, Any]] = {}
task_lock = Lock()

class PaperGenerationRequest(BaseModel):
    """Request model for paper generation"""
    user_query: str = Field(..., description="User's research query")
    user_name: str = Field(default="researcher", description="Username")
    task_id: Optional[str] = Field(default=None, description="Custom task ID")
    output_dir: str = Field(default="temp", description="Output directory")
    rag_service_url: Optional[str] = Field(default=None, description="RAG service URL")

    # Pipeline configuration
    outline_max_reflections: int = Field(default=1, description="Max outline reflections")
    outline_max_sections: int = Field(default=5, description="Max sections in outline")
    outline_min_depth: int = Field(default=1, description="Min outline depth")
    section_writer_model: str = Field(default="Qwen3-8B", description="Model for section writing")
    do_section_reflection: bool = Field(default=True, description="Enable section reflection")
    section_reflection_max_turns: int = Field(default=1, description="Max section reflection turns")
    do_global_reflection: bool = Field(default=True, description="Enable global reflection")
    global_reflection_max_turns: int = Field(default=1, description="Max global reflection turns")
    global_abstract_conclusion_max_turns: int = Field(default=1, description="Max abstract/conclusion turns")
    do_query_understand: bool = Field(default=True, description="Enable query understanding")
    only_outline: bool = Field(default=False, description="Generate only outline")

class TaskStatusResponse(BaseModel):
    """Response model for task status"""
    task_id: str
    status: str  # "running", "completed", "error", "not_found"
    progress: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str

async def run_paper_generation_task(
    request: PaperGenerationRequest,
    task_id: str
) -> Dict[str, Any]:
    """Background task for paper generation"""
    try:
        # Update task status
        with task_lock:
            active_tasks[task_id]["status"] = "running"
            active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        logger.info(f"Starting paper generation for task {task_id}")
        start_time = time.time()

        # Prepare kwargs for pipeline
        kwargs = {
            "outline_max_reflections": request.outline_max_reflections,
            "outline_max_sections": request.outline_max_sections,
            "outline_min_depth": request.outline_min_depth,
            "section_writer_model": request.section_writer_model,
            "do_section_reflection": request.do_section_reflection,
            "section_reflection_max_turns": request.section_reflection_max_turns,
            "do_global_reflection": request.do_global_reflection,
            "global_reflection_max_turns": request.global_reflection_max_turns,
            "global_abstract_conclusion_max_turns": request.global_abstract_conclusion_max_turns,
            "do_query_understand": request.do_query_understand,
            "only_outline": request.only_outline,
        }

        # Create pipeline
        pipeline = PaperGenerationPipeline(
            user_name=request.user_name,
            user_query=request.user_query,
            task_id=task_id,
            output_dir=request.output_dir,
            rag_service_url=request.rag_service_url,
            **kwargs
        )

        # Generate paper
        results = await pipeline.generate_paper()

        # Calculate duration
        duration = time.time() - start_time
        minutes, seconds = divmod(duration, 60)

        # Save results
        output_path = save_results(results, request.output_dir)

        # Update task status
        with task_lock:
            active_tasks[task_id]["status"] = "completed"
            active_tasks[task_id]["result"] = {
                "paper_title": results.get("paper_title", ""),
                "output_path": output_path,
                "duration_minutes": int(minutes),
                "duration_seconds": int(seconds),
                "has_error": "error" in results
            }
            active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            active_tasks[task_id]["full_results"] = results

        logger.info(f"Task {task_id} completed successfully in {int(minutes)}m {int(seconds)}s")
        return results

    except Exception as e:
        error_msg = f"Task {task_id} failed: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")

        # Update task status with error
        with task_lock:
            active_tasks[task_id]["status"] = "error"
            active_tasks[task_id]["error"] = error_msg
            active_tasks[task_id]["updated_at"] = datetime.now().isoformat()

        return {"error": error_msg}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page"""
    html_file_path = os.path.join(static_dir, "index.html")

    # 如果 index.html 不存在，返回简单的重定向页面
    if not os.path.exists(html_file_path):
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Academic Paper Generation API</title>
        </head>
        <body>
            <h1>Academic Paper Generation API</h1>
            <p>请将 index.html 文件放在 static/ 目录下</p>
            <p>API 文档: <a href="/docs">/docs</a></p>
        </body>
        </html>
        """)

    # 读取并返回 index.html
    with open(html_file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)


# 添加临时调试端点

@app.get("/api/debug-request")
async def debug_request(
    user_query: str = None,
    user_name: str = "researcher",
    output_dir: str = "temp",
    outline_max_sections: int = 5,
    section_writer_model: str = "Qwen3-32B",
    outline_max_reflections: int = 1,
    do_section_reflection: bool = True,
    do_global_reflection: bool = True,
    do_query_understand: bool = True,
    only_outline: bool = False
):
    """Debug endpoint to see what's being received"""
    logger.info("Debug endpoint called with parameters:")
    params = {
        "user_query": user_query,
        "user_name": user_name,
        "output_dir": output_dir,
        "outline_max_sections": outline_max_sections,
        "section_writer_model": section_writer_model,
        "outline_max_reflections": outline_max_reflections,
        "do_section_reflection": do_section_reflection,
        "do_global_reflection": do_global_reflection,
        "do_query_understand": do_query_understand,
        "only_outline": only_outline
    }
    logger.info(f"Parameters: {params}")
    return {"message": "Debug endpoint reached", "parameters": params}

@app.post("/api/generate_paper", response_model=Dict[str, str])
async def generate_paper(
    request: PaperGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Start paper generation task"""
    try:
        logger.info(f"user request come in: {request}")
        # Generate task ID
        task_id = request.task_id or f"{request.user_name}_{str(uuid.uuid4())[:8]}"

        # Check if task already exists
        with task_lock:
            if task_id in active_tasks:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task with ID {task_id} already exists"
                )

            # Initialize task
            active_tasks[task_id] = {
                "task_id": task_id,
                "status": "initializing",
                "progress": {},
                "result": None,
                "error": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "request": request.dict()
            }

        # Add background task
        background_tasks.add_task(
            run_paper_generation_task,
            request,
            task_id
        )

        logger.info(f"Started paper generation task {task_id} for query: {request.user_query}")

        return {
            "task_id": task_id,
            "status": "started",
            "message": "Paper generation task started successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start paper generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start paper generation: {str(e)}"
        )


@app.post("/api/update-env")
async def update_environment_variables(env_vars: Dict[str, str]):
    """Update environment variables"""
    try:
        for key, value in env_vars.items():
            if key in ["GOOGLE_SERPER_KEY", "SERPAPI_API_KEY"]:
                if value:  # Only set if value is not empty
                    os.environ[key] = value
                    logger.info(f"Updated environment variable: {key}")
                else:
                    # Remove from environment if empty
                    if key in os.environ:
                        del os.environ[key]
                        logger.info(f"Removed environment variable: {key}")

        return {"message": "Environment variables updated successfully"}

    except Exception as e:
        logger.error(f"Failed to update environment variables: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update environment variables: {str(e)}"
        )


@app.get("/api/task-result/{task_id}")
async def get_task_result(task_id: str):
    """Get complete task result"""
    try:
        with task_lock:
            if task_id not in active_tasks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {task_id} not found"
                )

            task_data = active_tasks[task_id]

            if task_data["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Task {task_id} is not completed yet. Status: {task_data['status']}"
                )

            return task_data.get("full_results", {})

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task result: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task result: {str(e)}"
        )



@app.get("/api/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get task status and progress"""
    try:
        with task_lock:
            if task_id not in active_tasks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {task_id} not found"
                )

            task_data = active_tasks[task_id].copy()

        return TaskStatusResponse(
            task_id=task_data["task_id"],
            status=task_data["status"],
            progress=task_data["progress"],
            result=task_data.get("result"),
            error=task_data.get("error"),
            created_at=task_data["created_at"],
            updated_at=task_data["updated_at"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task status: {str(e)}"
        )

@app.get("/api/tasks")
async def list_tasks():
    """List all tasks"""
    try:
        with task_lock:
            tasks = []
            for task_id, task_data in active_tasks.items():
                tasks.append({
                    "task_id": task_id,
                    "status": task_data["status"],
                    "user_query": task_data.get("request", {}).get("user_query", ""),
                    "created_at": task_data["created_at"],
                    "updated_at": task_data["updated_at"],
                    "error": task_data.get("error")
                })

        return tasks  # Return the tasks array directly, not wrapped in a dict

    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list tasks: {str(e)}"
        )

@app.get("/api/download/{task_id}")
async def download_paper(task_id: str, file_type: str = "markdown"):
    """Download generated paper file"""
    try:
        with task_lock:
            if task_id not in active_tasks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {task_id} not found"
                )

            task_data = active_tasks[task_id]

            if task_data["status"] != "completed":
                raise HTTPException(
                    status_code=400,
                    detail=f"Task {task_id} is not completed yet"
                )

            result = task_data.get("result", {})
            output_path = result.get("output_path", "")
            if not output_path:
                raise HTTPException(
                    status_code=404,
                    detail="Output file not found"
                )

        # Determine file path based on type
        base_path = output_path.replace("_process.json", "")

        if file_type == "markdown":
            file_path = f"{base_path}_process.md"
            media_type = "text/markdown"
        elif file_type == "json":
            file_path = f"{base_path}_paper.json"
            media_type = "application/json"
        elif file_type == "process":
            file_path = output_path
            media_type = "application/json"
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Choose from: markdown, json, process"
            )

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File {file_path} not found"
            )

        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=os.path.basename(file_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download file: {str(e)}"
        )

# Add missing cancel task endpoint
@app.post("/api/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        with task_lock:
            if task_id not in active_tasks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {task_id} not found"
                )

            task_data = active_tasks[task_id]
            if task_data["status"] not in ["running", "initializing"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Task {task_id} cannot be cancelled (status: {task_data['status']})"
                )

            # Update task status to cancelled
            active_tasks[task_id]["status"] = "cancelled"
            active_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            active_tasks[task_id]["error"] = "Task cancelled by user"

        return {"message": f"Task {task_id} has been cancelled"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task"""
    try:
        with task_lock:
            if task_id not in active_tasks:
                raise HTTPException(
                    status_code=404,
                    detail=f"Task {task_id} not found"
                )

            del active_tasks[task_id]

        return {"message": f"Task {task_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete task: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete task: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": len(active_tasks)
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Academic Paper Generation API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn

    # Create output directory
    os.makedirs("temp", exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)  # 确保静态文件目录存在

    logger.info("Starting Academic Paper Generation Server...")
    logger.info(f"Frontend will be available at: http://localhost:8193/")
    logger.info(f"API documentation at: http://localhost:8193/docs")

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8193,
        log_level="info",
        reload=False  # Set to True for development
    )