from fastapi import APIRouter
from services import info, event_bus

router = APIRouter(prefix="/info", tags=["info"])

API_VERSION = "0.1.0"


@router.get("")
async def full_info():
    """Full system snapshot — CPU, RAM, disk, temp, uptime, and active sessions."""
    stats = info.get_all()
    return {
        "api": {
            "version": API_VERSION,
            "active_sessions": len(event_bus._buses),
        },
        "platform": {
            "os": stats["platform"],
            "machine": stats["machine"],
            "python_version": stats["python_version"],
            "hostname": stats["hostname"],
        },
        "cpu": {
            "percent": stats["cpu_percent"],
            "temp_c": stats["cpu_temp_c"],
        },
        "memory": stats["memory"],
        "disk": stats["disk"],
        "uptime": {
            "process_sec": stats["process_uptime_sec"],
            "system_sec": stats["system_uptime_sec"],
        },
    }


@router.get("/health")
async def health():
    """Lightweight liveness check — no hardware reads."""
    return {"status": "ok", "version": API_VERSION}


@router.get("/system")
async def system():
    """Hardware stats only — CPU, RAM, disk, temperature."""
    stats = info.get_all()
    return {
        "cpu": {
            "percent": stats["cpu_percent"],
            "temp_c": stats["cpu_temp_c"],
        },
        "memory": stats["memory"],
        "disk": stats["disk"],
        "uptime": {
            "process_sec": stats["process_uptime_sec"],
            "system_sec": stats["system_uptime_sec"],
        },
    }
