from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, ForeignKey, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)

from sqlalchemy.dialects.postgresql import JSONB

from sqlalchemy.orm import relationship, declarative_base
from core.db import Base



class Instrument(Base):
    __tablename__ = "instruments"
    
    id = Column(Integer, primary_key=True)  # primary_key is already indexed by default
    composite_figi = Column(String(20), nullable=False) 
    composite_figi_hash = Column(BigInteger, index=True, nullable=True) # Changed to BigInteger for safety
    
    ticker = Column(String(20)) 
    ticker_hash = Column(BigInteger, nullable=True)
    name = Column(String(255)) 
     
    cik = Column(BigInteger, nullable=True)
    
    sic_code = Column(Integer, nullable=True)
    sic_description = Column(String(255), nullable=True)
    total_employees = Column(Integer, nullable=True)
    market_cap = Column(BigInteger, nullable=True)  # Safely holds trillions in dollar value
    share_class_shares_outstanding = Column(BigInteger, nullable=True)
    weighted_shares_outstanding = Column(BigInteger, nullable=True)
    city = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)  # Handles international alpha codes too
    state = Column(String(50), nullable=True)        # Accommodates full names or region codes


    market = Column(String(20)) 
    locale = Column(String(20)) 
    primary_exchange = Column(String(20))
    type = Column(String(20))
    active = Column(Boolean, default=True) 
    currency_name = Column(String(20)) 
    share_class_figi = Column(String(20))
    point_in_time_date = Column(Date)
    upsert_date = Column(Date)



    __table_args__ = (
        Index('idx_figi_hash_pit', 'composite_figi_hash', 'point_in_time_date'),
        UniqueConstraint('ticker', 'point_in_time_date', name='uq_instruments_ticker_pit'),
    )