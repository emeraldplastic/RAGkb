"""
Audit Logging Service for RAG KB.
Comprehensive audit logging for security, compliance, and debugging.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging
import json
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events."""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    DOCUMENT_UPLOAD = "document_upload"
    DOCUMENT_DELETE = "document_delete"
    SEARCH_QUERY = "search_query"
    API_ACCESS = "api_access"
    CONFIG_CHANGE = "config_change"
    ERROR_OCCURRED = "error_occurred"
    SECURITY_EVENT = "security_event"

class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Represents an audit event."""
    event_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    session_id: Optional[str]
    action: str
    resource: Optional[str]
    details: Dict[str, Any]
    status: str
    duration_ms: Optional[int]

@dataclass
class AuditConfig:
    """Configuration for audit logging."""
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    log_file_path: str = "audit.log"
    max_log_size_mb: int = 100
    retention_days: int = 90
    log_sensitive_data: bool = False
    enable_performance_logging: bool = True

class AuditLoggingService:
    """Service for comprehensive audit logging."""
    
    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        self.audit_log: List[AuditEvent] = []
        self.event_counter = 0
        
        # Setup file logging if enabled
        if self.config.enable_file_logging:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file logging for audit events."""
        file_handler = logging.FileHandler(self.config.log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        audit_logger = logging.getLogger('audit')
        audit_logger.addHandler(file_handler)
        audit_logger.setLevel(logging.INFO)
    
    def log_event(
        self,
        event_type: AuditEventType,
        action: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        status: str = "success",
        duration_ms: Optional[int] = None
    ) -> AuditEvent:
        """
        Log an audit event.
        
        Args:
            event_type: Type of audit event
            action: Description of the action
            user_id: User identifier
            ip_address: IP address of the request
            session_id: Session identifier
            resource: Resource being accessed
            details: Additional event details
            severity: Severity level
            status: Status of the action
            duration_ms: Duration of the action in milliseconds
        
        Returns:
            Created AuditEvent
        """
        self.event_counter += 1
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.event_counter}"
        
        # Filter sensitive data if configured
        if not self.config.log_sensitive_data and details:
            details = self._filter_sensitive_data(details)
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            user_id=user_id,
            ip_address=ip_address,
            session_id=session_id,
            action=action,
            resource=resource,
            details=details or {},
            status=status,
            duration_ms=duration_ms
        )
        
        self.audit_log.append(event)
        
        # Log to console if enabled
        if self.config.enable_console_logging:
            self._log_to_console(event)
        
        # Log to file if enabled
        if self.config.enable_file_logging:
            self._log_to_file(event)
        
        return event
    
    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out sensitive data from logs."""
        sensitive_keys = {
            'password', 'token', 'api_key', 'secret', 'credit_card',
            'ssn', 'social_security', 'auth', 'credential'
        }
        
        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        
        return filtered
    
    def _log_to_console(self, event: AuditEvent):
        """Log event to console."""
        log_message = f"[{event.severity.value.upper()}] {event.event_type.value}: {event.action}"
        
        if event.user_id:
            log_message += f" | User: {event.user_id}"
        
        if event.resource:
            log_message += f" | Resource: {event.resource}"
        
        if event.status != "success":
            log_message += f" | Status: {event.status}"
        
        if event.duration_ms:
            log_message += f" | Duration: {event.duration_ms}ms"
        
        if event.severity == AuditSeverity.CRITICAL:
            logger.critical(log_message)
        elif event.severity == AuditSeverity.ERROR:
            logger.error(log_message)
        elif event.severity == AuditSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _log_to_file(self, event: AuditEvent):
        """Log event to file."""
        audit_logger = logging.getLogger('audit')
        
        log_data = {
            'event_id': event.event_id,
            'event_type': event.event_type.value,
            'severity': event.severity.value,
            'timestamp': event.timestamp.isoformat(),
            'user_id': event.user_id,
            'ip_address': event.ip_address,
            'session_id': event.session_id,
            'action': event.action,
            'resource': event.resource,
            'details': event.details,
            'status': event.status,
            'duration_ms': event.duration_ms
        }
        
        audit_logger.info(json.dumps(log_data))
    
    def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100
    ) -> List[AuditEvent]:
        """
        Retrieve audit events with filtering.
        
        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            start_date: Filter events after this date
            end_date: Filter events before this date
            severity: Filter by severity
            limit: Maximum number of events to return
        
        Returns:
            List of filtered AuditEvent objects
        """
        filtered = self.audit_log
        
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id]
        
        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]
        
        if severity:
            filtered = [e for e in filtered if e.severity == severity]
        
        # Sort by timestamp descending
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered[:limit]
    
    def get_user_activity(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """
        Get activity summary for a specific user.
        
        Args:
            user_id: User identifier
            days: Number of days to look back
        
        Returns:
            Dictionary with user activity statistics
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        user_events = [
            e for e in self.audit_log
            if e.user_id == user_id and e.timestamp >= cutoff_date
        ]
        
        # Count events by type
        event_counts = {}
        for event in user_events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
        
        # Calculate average duration
        durations = [e.duration_ms for e in user_events if e.duration_ms]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Count errors
        error_count = sum(1 for e in user_events if e.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL])
        
        return {
            'user_id': user_id,
            'period_days': days,
            'total_events': len(user_events),
            'events_by_type': event_counts,
            'average_duration_ms': avg_duration,
            'error_count': error_count,
            'success_rate': (len(user_events) - error_count) / len(user_events) if user_events else 0
        }
    
    def get_security_events(self, hours: int = 24) -> List[AuditEvent]:
        """
        Get security-related events.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            List of security-related AuditEvent objects
        """
        cutoff_date = datetime.now() - timedelta(hours=hours)
        
        security_events = [
            e for e in self.audit_log
            if e.event_type in [AuditEventType.SECURITY_EVENT, AuditEventType.USER_LOGIN, AuditEventType.USER_LOGOUT]
            and e.timestamp >= cutoff_date
        ]
        
        security_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return security_events
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of error events.
        
        Args:
            hours: Number of hours to look back
        
        Returns:
            Dictionary with error statistics
        """
        cutoff_date = datetime.now() - timedelta(hours=hours)
        
        error_events = [
            e for e in self.audit_log
            if e.severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]
            and e.timestamp >= cutoff_date
        ]
        
        # Group by action
        error_by_action = {}
        for event in error_events:
            error_by_action[event.action] = error_by_action.get(event.action, 0) + 1
        
        # Group by user
        error_by_user = {}
        for event in error_events:
            if event.user_id:
                error_by_user[event.user_id] = error_by_user.get(event.user_id, 0) + 1
        
        return {
            'period_hours': hours,
            'total_errors': len(error_events),
            'errors_by_action': error_by_action,
            'errors_by_user': error_by_user,
            'critical_errors': sum(1 for e in error_events if e.severity == AuditSeverity.CRITICAL)
        }
    
    def cleanup_old_events(self):
        """Remove events older than retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        original_count = len(self.audit_log)
        self.audit_log = [e for e in self.audit_log if e.timestamp >= cutoff_date]
        
        removed_count = original_count - len(self.audit_log)
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old audit events")
    
    def export_audit_log(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json"
    ) -> str:
        """
        Export audit log in specified format.
        
        Args:
            start_date: Start of export period
            end_date: End of export period
            format: Export format (json, csv)
        
        Returns:
            Exported audit log as string
        """
        events = self.get_events(start_date=start_date, end_date=end_date, limit=10000)
        
        if format == "json":
            export_data = [
                {
                    'event_id': e.event_id,
                    'event_type': e.event_type.value,
                    'severity': e.severity.value,
                    'timestamp': e.timestamp.isoformat(),
                    'user_id': e.user_id,
                    'ip_address': e.ip_address,
                    'session_id': e.session_id,
                    'action': e.action,
                    'resource': e.resource,
                    'details': e.details,
                    'status': e.status,
                    'duration_ms': e.duration_ms
                }
                for e in events
            ]
            return json.dumps(export_data, indent=2)
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            writer.writerow([
                'event_id', 'event_type', 'severity', 'timestamp', 'user_id',
                'ip_address', 'session_id', 'action', 'resource', 'status', 'duration_ms'
            ])
            
            for e in events:
                writer.writerow([
                    e.event_id, e.event_type.value, e.severity.value,
                    e.timestamp.isoformat(), e.user_id or '', e.ip_address or '',
                    e.session_id or '', e.action, e.resource or '',
                    e.status, e.duration_ms or ''
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def audit_decorator(
        self,
        event_type: AuditEventType,
        action: str,
        severity: AuditSeverity = AuditSeverity.INFO
    ):
        """
        Decorator for automatic audit logging of function calls.
        
        Args:
            event_type: Type of audit event
            action: Description of the action
            severity: Severity level
        
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = datetime.now()
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    # Extract user_id from kwargs if available
                    user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], str) else None)
                    
                    self.log_event(
                        event_type=event_type,
                        action=action,
                        user_id=user_id,
                        status="success",
                        duration_ms=duration_ms,
                        severity=severity
                    )
                    
                    return result
                    
                except Exception as e:
                    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    user_id = kwargs.get('user_id') or (args[0] if args and isinstance(args[0], str) else None)
                    
                    self.log_event(
                        event_type=AuditEventType.ERROR_OCCURRED,
                        action=f"{action} - Error",
                        user_id=user_id,
                        details={'error': str(e)},
                        status="error",
                        duration_ms=duration_ms,
                        severity=AuditSeverity.ERROR
                    )
                    
                    raise
            
            return wrapper
        return decorator

# Global audit logging service instance
audit_logging_service = AuditLoggingService()

def test_audit_logging():
    """Test the audit logging service."""
    service = AuditLoggingService()
    
    # Test event logging
    event = service.log_event(
        event_type=AuditEventType.USER_LOGIN,
        action="User logged in",
        user_id="user123",
        ip_address="192.168.1.1",
        session_id="session456",
        details={"login_method": "password"}
    )
    
    print(f"Logged event: {event.event_id}")
    
    # Test error logging
    error_event = service.log_event(
        event_type=AuditEventType.ERROR_OCCURRED,
        action="Database connection failed",
        severity=AuditSeverity.ERROR,
        details={"error": "Connection timeout"}
    )
    
    print(f"Logged error: {error_event.event_id}")
    
    # Get user activity
    activity = service.get_user_activity("user123", days=1)
    print(f"User activity: {activity}")
    
    # Get security events
    security_events = service.get_security_events(hours=24)
    print(f"Security events: {len(security_events)}")

if __name__ == "__main__":
    test_audit_logging()
