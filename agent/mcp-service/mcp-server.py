import pymysql
from fastmcp import FastMCP
from dotenv import load_dotenv
import threading
from tools import run_task_process, get_db_connection
import os

# MCP Service
load_dotenv()

mcp = FastMCP("NewsServer")

@mcp.tool()
def get_tasks():
    """View all training tasks created by the user"""
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM task"
            cursor.execute(sql)
            tasks = cursor.fetchall()
            return {
                "status": "success",
                "total": len(tasks),
                "tasks": tasks
            }
    finally:
        conn.close()

@mcp.tool()
def get_datasets():
    """View all datasets uploaded by the user"""
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM dataset"
            cursor.execute(sql)
            datasets = cursor.fetchall()
            return {
                "status": "success",
                "total": len(datasets),
                "datasets": datasets
            }
    finally:
        conn.close()
import os

@mcp.tool()
def create_task(dataset_id: int):
    """Create a new task
    Use ScLinformer to create an analysis task
    Includes data preprocessing, model training, and test set evaluation! ✨
    Args:
        dataset_id: Which dataset to use - if there's only one, just use it; if you say "the one just now" or "that one" I'll understand; only ask if you're unsure
    My rules:
    - Only use ScLinformer, for other models I'll say "Not supported yet, stay tuned~"
    - If you say train/analyze/run, I'll create the task immediately!
    - I won't ask "which model to use", ScLinformer is the only choice 💫
    """
    description = "scLinformer training and testing all-in-one! ✨"
    # Fixed folder path
    base_task_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_data", "task")
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # First insert task record to get task ID
            sql = """
            INSERT INTO task (description, dataset_id)
            VALUES (%s, %s)
            """
            cursor.execute(sql, (description, dataset_id))
            conn.commit()
            task_id = cursor.lastrowid
            
            # Create dedicated folder
            task_folder = os.path.join(base_task_folder, str(task_id))
            os.makedirs(task_folder, exist_ok=True)
            
            # Update task path information
            model_weight_path = os.path.join(task_folder, "model")
            test_result_path = task_folder
            
            # Create subfolders
            os.makedirs(model_weight_path, exist_ok=True)
            os.makedirs(test_result_path, exist_ok=True)
            
            # Update database record
            update_sql = """
            UPDATE task SET model_weight_path = %s, test_result_path = %s, start_time = NOW()
            WHERE id = %s
            """
            cursor.execute(update_sql, (model_weight_path, test_result_path, task_id))
            conn.commit()

            thread = threading.Thread(
                target=run_task_process,
                args=(task_id, dataset_id, task_folder),
                daemon=True  # Daemon thread, exits automatically when main program exits
            )
            thread.start()
            
            return {
                "status": "success",
                "task_id": task_id,
                "message": "Task created successfully",
                "task_info": {
                    "description": description,
                    "dataset_id": dataset_id,
                    "model_weight_path": model_weight_path,
                    "test_result_path": test_result_path
                }
            }
    finally:
        conn.close()

@mcp.tool()
def check_task_status(task_id: int):
    """Check if the training task is completed
    If task is completed, return the results
    Task results may contain image paths: umap_cell_path and umap_batch_path
    If present, return the image paths in the following format:
        <img>umap_cell_path</img>
        <img>umap_batch_path</img>
    No other requirements, please keep your cute style 😊
    Args:
        task_id: Task ID
    """
    conn = get_db_connection()
    try:
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = "SELECT * FROM task WHERE id = %s"
            cursor.execute(sql, (task_id,))
            task = cursor.fetchone()
            if not task:
                return {
                    "status": "error",
                    "message": f"Task ID {task_id} does not exist"
                }
            if task['complete_time']:
                return {
                    "status": "success",
                    "task_status": "completed",
                    "task_info": task
                }
            else:
                return {
                    "status": "success",
                    "task_status": "pending",
                    "message": f"Task ID {task_id} is not yet completed",
                    "task_info": task
                }
    finally:
        conn.close()

# Image server
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

IMAGE_SERVER_PORT = 8001
IMAGE_SERVER_URL = f"http://localhost:{IMAGE_SERVER_PORT}/static"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DATA_DIR = os.path.join(BASE_DIR, "runtime_data")

image_app = FastAPI()
image_app.mount("/static", StaticFiles(directory=RUNTIME_DATA_DIR), name="static")

def start_image_server():
    print(f"Image server: http://localhost:{IMAGE_SERVER_PORT}")
    uvicorn.run(image_app, host="0.0.0.0", port=IMAGE_SERVER_PORT, log_level="warning")

if __name__ == "__main__":
    image_thread = threading.Thread(target=start_image_server, daemon=True)
    image_thread.start()
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
