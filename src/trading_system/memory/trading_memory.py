"""Trading memory and learning system with checkpointing."""

import json
import sqlite3
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    trade_id: str
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    direction: str  # "long" or "short"
    entry_time: datetime
    exit_time: Optional[datetime]
    profit_loss: Optional[float]
    profit_loss_pct: Optional[float]
    reason_opened: str
    reason_closed: Optional[str]
    agent_signals: Dict[str, Any]
    portfolio_metrics: Dict[str, float]
    market_conditions: Dict[str, Any]
    lessons_learned: str = ""
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['entry_time'] = self.entry_time.isoformat()
        d['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        return d


@dataclass
class Checkpoint:
    """System checkpoint for recovery."""
    checkpoint_id: str
    timestamp: datetime
    layer: int  # Which layer completed
    state_data: Dict[str, Any]
    signals: List[Dict[str, Any]]
    decision: Optional[str]
    hash_value: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class LessonLearned:
    """Extracted lesson from trade outcome."""
    lesson_id: str
    trade_id: str
    category: str  # "success" or "failure"
    description: str
    pattern: Dict[str, Any]
    applicable_symbols: List[str]
    applicable_conditions: Dict[str, Any]
    confidence: float  # 0-1
    created_at: datetime = field(default_factory=datetime.utcnow)
    usage_count: int = 0
    success_rate: float = 1.0


class TradingMemory:
    """Persistent trading memory with SQLite."""
    
    def __init__(self, db_path: str = "./data/trading_memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                position_size REAL,
                direction TEXT,
                entry_time TEXT,
                exit_time TEXT,
                profit_loss REAL,
                profit_loss_pct REAL,
                reason_opened TEXT,
                reason_closed TEXT,
                agent_signals TEXT,
                portfolio_metrics TEXT,
                market_conditions TEXT,
                lessons_learned TEXT,
                success BOOLEAN,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Lessons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lessons (
                lesson_id TEXT PRIMARY KEY,
                trade_id TEXT,
                category TEXT,
                description TEXT,
                pattern TEXT,
                applicable_symbols TEXT,
                applicable_conditions TEXT,
                confidence REAL,
                created_at TEXT,
                usage_count INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 1.0,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)
        
        # Checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                timestamp TEXT,
                layer INTEGER,
                state_data TEXT,
                signals TEXT,
                decision TEXT,
                hash_value TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record_trade(self, trade: TradeRecord) -> bool:
        """Record a completed trade."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            trade_dict = trade.to_dict()
            placeholders = ', '.join(['?' for _ in trade_dict])
            columns = ', '.join(trade_dict.keys())
            
            cursor.execute(
                f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
                tuple(trade_dict.values())
            )
            
            conn.commit()
            conn.close()
            logger.info(f"Trade recorded: {trade.trade_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to record trade: {e}")
            return False
    
    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 50) -> List[TradeRecord]:
        """Retrieve recent trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute(
                    "SELECT * FROM trades WHERE symbol = ? ORDER BY entry_time DESC LIMIT ?",
                    (symbol, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                    (limit,)
                )
            
            rows = cursor.fetchall()
            conn.close()
            
            trades = []
            for row in rows:
                # Reconstruct TradeRecord from database row
                trades.append(row)
            
            return trades
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}")
            return []
    
    def record_lesson(self, lesson: LessonLearned) -> bool:
        """Record a lesson learned from a trade."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO lessons 
                (lesson_id, trade_id, category, description, pattern, applicable_symbols, 
                 applicable_conditions, confidence, created_at, usage_count, success_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                lesson.lesson_id,
                lesson.trade_id,
                lesson.category,
                lesson.description,
                json.dumps(lesson.pattern),
                json.dumps(lesson.applicable_symbols),
                json.dumps(lesson.applicable_conditions),
                lesson.confidence,
                lesson.created_at.isoformat(),
                lesson.usage_count,
                lesson.success_rate,
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Lesson recorded: {lesson.lesson_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to record lesson: {e}")
            return False
    
    def get_applicable_lessons(self, symbol: str, market_conditions: Dict[str, Any]) -> List[LessonLearned]:
        """Retrieve lessons applicable to current conditions."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM lessons 
                WHERE confidence > 0.6 
                ORDER BY success_rate DESC, usage_count DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            lessons = []
            for row in rows:
                # Check if lesson applies to current symbol and conditions
                applicable_symbols = json.loads(row[5])
                if symbol in applicable_symbols or '*' in applicable_symbols:
                    lessons.append(row)
            
            return lessons
        except Exception as e:
            logger.error(f"Failed to retrieve lessons: {e}")
            return []
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Remove old records beyond retention period."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
            
            cursor.execute(
                "DELETE FROM trades WHERE created_at < ?",
                (cutoff_date,)
            )
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {deleted_count} old trade records")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup records: {e}")
            return 0


class CheckpointManager:
    """Manages system checkpoints for recovery."""
    
    def __init__(self, storage_path: str = "./data/checkpoints"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Save a checkpoint to disk."""
        try:
            # Calculate hash for integrity
            checkpoint_str = json.dumps(checkpoint.to_dict(), default=str)
            checkpoint.hash_value = hashlib.sha256(checkpoint_str.encode()).hexdigest()
            
            checkpoint_file = self.storage_path / f"checkpoint_{checkpoint.checkpoint_id}.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {checkpoint.checkpoint_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from disk."""
        try:
            checkpoint_file = self.storage_path / f"checkpoint_{checkpoint_id}.json"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Verify integrity
            stored_hash = data.pop('hash_value')
            calculated_hash = hashlib.sha256(
                json.dumps(data, default=str).encode()
            ).hexdigest()
            
            if stored_hash != calculated_hash:
                logger.warning(f"Checkpoint integrity check failed: {checkpoint_id}")
                return None
            
            logger.info(f"Checkpoint loaded: {checkpoint_id}")
            return Checkpoint(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_latest_checkpoint(self, layer: Optional[int] = None) -> Optional[Checkpoint]:
        """Get the latest checkpoint, optionally for a specific layer."""
        try:
            checkpoints = sorted(self.storage_path.glob("checkpoint_*.json"), 
                               key=lambda x: x.stat().st_mtime, reverse=True)
            
            for checkpoint_file in checkpoints:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                
                if layer is None or data.get('layer') == layer:
                    return Checkpoint(**data)
            
            return None
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    def cleanup_old_checkpoints(self, retention_days: int = 30) -> int:
        """Remove old checkpoints beyond retention period."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=retention_days)
            deleted_count = 0
            
            for checkpoint_file in self.storage_path.glob("checkpoint_*.json"):
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if file_time < cutoff_time:
                    checkpoint_file.unlink()
                    deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")
            return 0


class ReflectionEngine:
    """Extracts lessons and reflections from trades."""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_reflection(self, trade: TradeRecord) -> str:
        """
        Generate reflection/lesson from a completed trade.
        """
        prompt = f"""
        Analyze this trade and extract key lessons:
        
        Trade Details:
        - Symbol: {trade.symbol}
        - Direction: {trade.direction}
        - Entry Price: {trade.entry_price}
        - Exit Price: {trade.exit_price}
        - P/L: {trade.profit_loss_pct}%
        - Reason Opened: {trade.reason_opened}
        - Reason Closed: {trade.reason_closed}
        - Agent Signals: {json.dumps(trade.agent_signals, indent=2)}
        
        Provide:
        1. What worked in this trade
        2. What didn't work
        3. Key lesson to apply next time
        4. Pattern recognition
        
        Be concise and actionable.
        """
        
        try:
            reflection = await self.llm_client.generate(prompt, max_tokens=400)
            return reflection
        except Exception as e:
            logger.error(f"Failed to generate reflection: {e}")
            return f"Analysis failed: {str(e)}"
    
    async def inject_lessons_into_prompt(self, symbol: str, base_prompt: str, memory: TradingMemory) -> str:
        """
        Enhance a prompt with relevant lessons learned.
        """
        lessons = memory.get_applicable_lessons(symbol, {})
        
        if not lessons:
            return base_prompt
        
        lessons_text = "\n\nApplicable lessons from past trades:\n"
        for lesson in lessons[:5]:  # Top 5 most relevant lessons
            lessons_text += f"- {lesson[3]} (confidence: {lesson[7]})\n"
        
        return base_prompt + lessons_text
