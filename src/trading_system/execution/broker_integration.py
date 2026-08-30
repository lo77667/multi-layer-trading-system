"""Broker integration with safety mechanisms."""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # "buy" or "sell"
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Open position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: datetime


class KillSwitch:
    """Global kill switch for risk management."""
    
    def __init__(self, loss_threshold: float = 0.02, enabled: bool = True):
        self.loss_threshold = loss_threshold  # 2% by default
        self.enabled = enabled
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self.trigger_reason = ""
    
    def check(self, current_balance: float, initial_balance: float) -> bool:
        """
        Check if kill switch should be triggered.
        
        Returns:
            True if triggered, False otherwise
        """
        if not self.enabled or self.triggered:
            return self.triggered
        
        loss_pct = (initial_balance - current_balance) / initial_balance
        
        if loss_pct >= self.loss_threshold:
            self.trigger(f"Daily loss exceeded {self.loss_threshold*100}%")
            return True
        
        return False
    
    def trigger(self, reason: str = ""):
        """Manually trigger kill switch."""
        self.triggered = True
        self.trigger_time = datetime.utcnow()
        self.trigger_reason = reason
        logger.critical(f"⚠️  KILL SWITCH TRIGGERED: {reason}")
    
    def reset(self):
        """Reset kill switch (for next trading day)."""
        self.triggered = False
        self.trigger_time = None
        self.trigger_reason = ""
        logger.info("Kill switch reset")


class BrokerGateway:
    """Abstract broker interface."""
    
    def __init__(self, mode: str = "paper"):
        self.mode = mode  # "paper" or "live"
        self.paper_mode = mode == "paper"
        logger.info(f"Broker initialized in {mode} mode")
    
    async def place_order(self, order: Order) -> bool:
        """Place an order."""
        raise NotImplementedError()
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        raise NotImplementedError()
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        raise NotImplementedError()
    
    async def get_balance(self) -> float:
        """Get account balance."""
        raise NotImplementedError()
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        raise NotImplementedError()


class AlpacaBroker(BrokerGateway):
    """Alpaca broker integration."""
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        mode: str = "paper",
        base_url: Optional[str] = None,
    ):
        super().__init__(mode)
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url or (
            "https://paper-api.alpaca.markets" if self.paper_mode
            else "https://api.alpaca.markets"
        )
        self.orders: Dict[str, Order] = {}
        self.positions: List[Position] = []
        self.balance = 100000.0
    
    async def place_order(self, order: Order) -> bool:
        """
        Place an order (simulated or real).
        """
        if self.paper_mode:
            # Paper trading simulation
            self.orders[order.order_id] = order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = order.price or 100.0
            logger.info(f"[PAPER] Order placed: {order.order_id} - {order.symbol} {order.quantity}@{order.price}")
            return True
        else:
            # Real trading (requires actual API implementation)
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/v2/orders",
                        headers={"APCA-API-KEY-ID": self.api_key},
                        json={
                            "symbol": order.symbol,
                            "qty": order.quantity,
                            "side": order.side,
                            "type": order.order_type.value,
                            "time_in_force": "day",
                            "limit_price": order.price,
                            "stop_price": order.stop_price,
                        },
                        timeout=30.0,
                    )
                    if response.status_code in [200, 201]:
                        result = response.json()
                        order.order_id = result["id"]
                        self.orders[order.order_id] = order
                        logger.info(f"[LIVE] Order placed: {order.order_id} - {order.symbol}")
                        return True
                    else:
                        logger.error(f"Order placement failed: {response.text}")
                        return False
            except Exception as e:
                logger.error(f"Exception in place_order: {e}")
                return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.paper_mode:
            if order_id in self.orders:
                self.orders[order_id].status = OrderStatus.CANCELLED
                logger.info(f"[PAPER] Order cancelled: {order_id}")
                return True
            return False
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.delete(
                        f"{self.base_url}/v2/orders/{order_id}",
                        headers={"APCA-API-KEY-ID": self.api_key},
                        timeout=30.0,
                    )
                    if response.status_code == 204:
                        logger.info(f"[LIVE] Order cancelled: {order_id}")
                        return True
                    else:
                        logger.error(f"Cancel failed: {response.text}")
                        return False
            except Exception as e:
                logger.error(f"Exception in cancel_order: {e}")
                return False
    
    async def get_positions(self) -> List[Position]:
        """Get current positions."""
        if self.paper_mode:
            return self.positions
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/v2/positions",
                        headers={"APCA-API-KEY-ID": self.api_key},
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        positions_data = response.json()
                        positions = []
                        for pos in positions_data:
                            positions.append(Position(
                                symbol=pos["symbol"],
                                quantity=float(pos["qty"]),
                                entry_price=float(pos["avg_entry_price"]),
                                current_price=float(pos["current_price"]),
                                unrealized_pnl=float(pos["unrealized_pl"]),
                                unrealized_pnl_pct=float(pos["unrealized_plpc"]),
                                opened_at=datetime.fromisoformat(pos["created_at"]),
                            ))
                        return positions
                    else:
                        logger.error(f"Failed to get positions: {response.text}")
                        return []
            except Exception as e:
                logger.error(f"Exception in get_positions: {e}")
                return []
    
    async def get_balance(self) -> float:
        """Get account balance."""
        if self.paper_mode:
            return self.balance
        else:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{self.base_url}/v2/account",
                        headers={"APCA-API-KEY-ID": self.api_key},
                        timeout=30.0,
                    )
                    if response.status_code == 200:
                        account = response.json()
                        return float(account["cash"])
                    else:
                        logger.error(f"Failed to get balance: {response.text}")
                        return 0.0
            except Exception as e:
                logger.error(f"Exception in get_balance: {e}")
                return 0.0
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status."""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED


class ExecutionMaster:
    """Execution layer with position sizing and order splitting."""
    
    def __init__(self, broker: BrokerGateway, risk_config: 'RiskConfig'):
        self.broker = broker
        self.risk_config = risk_config
        self.kill_switch = KillSwitch(
            loss_threshold=risk_config.kill_switch_loss_threshold,
            enabled=risk_config.kill_switch_enabled,
        )
        self.trade_log: List[Order] = []
    
    async def execute_trade(
        self,
        symbol: str,
        direction: str,  # "buy" or "sell"
        confidence: float,
        portfolio_value: float,
        current_price: float,
        atr: float,
    ) -> Optional[str]:
        """
        Execute a trade with position sizing and risk management.
        
        Returns:
            Order ID if successful, None otherwise
        """
        # Check kill switch
        if self.kill_switch.triggered:
            logger.warning("⚠️  Trading disabled - Kill switch is active")
            return None
        
        # Calculate position size based on risk
        position_size = self._calculate_position_size(
            portfolio_value, current_price, atr, confidence
        )
        
        if position_size == 0:
            logger.warning(f"Position size is 0 for {symbol}")
            return None
        
        # Split order into 3-4 tranches
        tranche_size = position_size / 3
        order_ids = []
        
        for i in range(3):
            order = Order(
                order_id=f"{symbol}_{datetime.utcnow().timestamp()}_{i}",
                symbol=symbol,
                order_type=OrderType.LIMIT,
                side=direction,
                quantity=tranche_size,
                price=current_price * (1.001 if direction == "buy" else 0.999),  # Small slippage buffer
            )
            
            success = await self.broker.place_order(order)
            if success:
                order_ids.append(order.order_id)
                self.trade_log.append(order)
                logger.info(f"Tranche {i+1} executed for {symbol}: {tranche_size} @ {order.price}")
            else:
                logger.error(f"Failed to execute tranche {i+1}")
        
        return order_ids[0] if order_ids else None
    
    def _calculate_position_size(
        self,
        portfolio_value: float,
        current_price: float,
        atr: float,
        confidence: float,
    ) -> float:
        """
        Calculate position size using risk management rules.
        
        Position Size = (Portfolio Value * Risk Per Trade) / (ATR * Multiplier)
        """
        risk_amount = portfolio_value * self.risk_config.per_trade_risk
        stop_distance = atr * self.risk_config.atr_multiplier
        
        if stop_distance == 0:
            return 0.0
        
        position_size = risk_amount / stop_distance
        
        # Apply max position limit
        max_position = portfolio_value * self.risk_config.max_position_size
        max_units = max_position / current_price
        
        return min(position_size, max_units)
    
    async def close_position(self, symbol: str, position_size: float) -> bool:
        """
        Close an open position.
        """
        order = Order(
            order_id=f"close_{symbol}_{datetime.utcnow().timestamp()}",
            symbol=symbol,
            order_type=OrderType.MARKET,
            side="sell",
            quantity=position_size,
        )
        
        return await self.broker.place_order(order)


class RiskConfig:
    """Risk management configuration (placeholder for import)."""
    pass
