from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, ForeignKey, text, DateTime, String,  Text,
    JSON, Boolean, Numeric # Import the JSON type
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB


Base = declarative_base()

class Company(Base):
    __tablename__ = "companies"
    id     = Column(Integer, primary_key=True)
    ticker = Column(String(10), unique=True, index=True, nullable=False)

    quotes      = relationship("Quote", back_populates="company")
    subsecond_quotes = relationship("SubsecondQuote", back_populates="company") 
    historical_emulate_quotes = relationship("HistoricalEmulateQuote", back_populates="company") 
    option_quotes = relationship("OptionQuote", back_populates="company") 
    eod_option_quotes = relationship("EODOptionQuote", back_populates="company") 
    theory_option_quotes = relationship("TheoryOptionQuote", back_populates="company") 

class Quote(Base):
    __tablename__ = "quotes"

    id             = Column(BigInteger, primary_key=True)
    company_id     = Column(Integer, ForeignKey("companies.id"),
                              nullable=False, index=True)
    # point the FK to time_entries.unix_ts
    time_entry_ts  = Column(BigInteger,
                             nullable=False, index=True)

    open_price     = Column(Float)
    high_price     = Column(Float)
    low_price      = Column(Float)
    close_price    = Column(Float)
    volume         = Column(BigInteger)

    company        = relationship("Company",   back_populates="quotes")

    __table_args__ = (
        UniqueConstraint("company_id", "time_entry_ts",
                         name="uq_company_timeentry"),
    )
class SubsecondQuote(Base): 
    __tablename__ = "subsecond_quotes"
    id = Column(BigInteger, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    time_entry_ts = Column(BigInteger, nullable=False, index=True) # Unix timestamp in microseconds/nanoseconds
    close_price = Column(Float)
    volume = Column(BigInteger)
    company = relationship("Company", back_populates="subsecond_quotes")

    __table_args__ = (
        UniqueConstraint("company_id", "time_entry_ts", name="uq_company_timeentry_subsecond"),
    )


class OptionQuote(Base):
    __tablename__ = "option_quotes"
    id = Column(BigInteger, primary_key=True)
    # should be derivable from the options string " O:SPY251219C00660000 " where spy is asset name, 251219 is expiry at 2025, dec 19. deconstruct this as dec
    option_name = Column(String(64), index=True, nullable=False)


    # either put or call C, P respectively
    contract_type = Column(String(4),  nullable=False)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False, index=True)
    company = relationship("Company", back_populates="option_quotes")

    contract_expiry = Column(BigInteger, nullable=False, index=True) # assume the effective expiry time is 4:00 EST at date. 251219 is expiry at 2025, dec 19. expressed in UNIX 
    strike      = Column(Float)

    # 2 step pipline fetch from api, upsert. 
    # derivable from body of polygon api https://api.polygon.io/stocks/v1/short-interest?limit=10&sort=ticker.asc&apiKey=demo
    time_entry_ts = Column(BigInteger, nullable=False, index=True)  # t  expressed in UNIX 
        # it is doable but 3 stage pipeline.  fetch from option api, map to quotes unix, upsert.
    underlying_close = Column(Float) #  at time_entry_ts of the underlying asset close NEEDS LOOKUP. 
    
    # it is doable but 3 stage pipeline.  fetch from option api, map to quotes unix, upsert.
    underlying_SOD = Column(Float) #  at time_entry_ts of the underlying asset close NEEDS LOOKUP. 
    
    option_open     = Column(Float) #o
    option_close     = Column(Float) #h
    option_high      = Column(Float) #l
    option_low    = Column(Float)  #c
    option_volume         = Column(BigInteger) #v
    option_volume_weighted  = Column(Float)  # vw
    option_transactions = Column(BigInteger) # n

    __table_args__ = (
        UniqueConstraint("option_name", "time_entry_ts", name="uq_contract_timeentry"),
    )

# https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey=demo
class FEDERAL_FUNDS_RATE(Base):
    __tablename__ = "federal_funds_rate"
    id = Column(BigInteger, primary_key=True)
    time_entry_ts = Column(BigInteger, nullable=False, index=True)  # t  expressed in UNIX 
 
    #daily follow the url, swap the demo key with the env key.
    risk_free_rate_r = Column(Float) 




    __table_args__ = (
        UniqueConstraint("time_entry_ts", name="uq_ffr_timeentry"),
    )


class TREASURY_YIELD(Base):
    __tablename__ = "treasury_yield"
    id = Column(BigInteger, primary_key=True)
    time_entry_ts = Column(BigInteger, nullable=False, index=True)  # t  expressed in UNIX 
    _3month = Column(Float)

    _2year =   Column(Float)
    _5year =   Column(Float)
    _7year =   Column(Float)
    _10year =   Column(Float)
    __table_args__ = (
        UniqueConstraint("time_entry_ts", name="uq_ty_timeentry"),
    )



# https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&time_from=20250901T0930&time_to=20251001T0930&limit=1000&apikey=JM19QBKIL3IFOLLT


class NEWS_SENTIMENT(Base):
    __tablename__ = "news_sentiment"
    id = Column(BigInteger, primary_key=True)

    title = Column(Text, nullable=False)
    url = Column(String(1024), nullable=False, index=True) # URLs can be long
    summary = Column(Text)
    

    time_published_ts = Column(BigInteger, nullable=False, index=True) # convert     "time_published": "20251001T072600", into unix
    source = Column(String(255))
    source_domain = Column(String(255))
    
    authors = Column(JSONB)
    topics = Column(JSONB)
    ticker_sentiment = Column(JSONB)
    
    # Overall sentiment for the article
    overall_sentiment_score = Column(Float)
    overall_sentiment_label = Column(String(50))

    __table_args__ = (
        UniqueConstraint("url", name="uq_article_url"),
    )




class Currency(Base):
    __tablename__ = "currencies"
    id = Column(Integer, primary_key=True)
    code = Column(String(3), unique=True, index=True, nullable=False) # e.g., 'USD'
    name = Column(String(50))
    is_liquidity_core = Column(Boolean, default=False) # For your 90% filter algorithm

    # Relationships to the edges
    rates_out = relationship("ForexRate", foreign_keys="ForexRate.from_currency_id", back_populates="from_currency")
    rates_in = relationship("ForexRate", foreign_keys="ForexRate.to_currency_id", back_populates="to_currency")

class ForexRate(Base):
    """
    This represents the 'Edge' in your Forex Graph.
    It stores the transaction cost (rate) from one currency to another over time.
    """
    __tablename__ = "forex_rates"

    id = Column(BigInteger, primary_key=True)
    from_currency_id = Column(Integer, ForeignKey("currencies.id"), nullable=False, index=True)
    to_currency_id   = Column(Integer, ForeignKey("currencies.id"), nullable=False, index=True)
    
    time_entry_ts = Column(BigInteger, nullable=False, index=True) # Unix timestamp
    
    # The Matrix Values
    rate = Column(Numeric(precision=20, scale=10), nullable=False) 
    bid  = Column(Numeric(precision=20, scale=10))
    ask  = Column(Numeric(precision=20, scale=10))

    from_currency = relationship("Currency", foreign_keys=[from_currency_id], back_populates="rates_out")
    to_currency   = relationship("Currency", foreign_keys=[to_currency_id], back_populates="rates_in")

    __table_args__ = (
        UniqueConstraint("from_currency_id", "to_currency_id", "time_entry_ts", name="uq_forex_path_time"),
    )





class TrainItem(Base):


    __tablename__ = "train_item"
    id            = Column(BigInteger, primary_key=True, autoincrement=True)
    time_entry_ts = Column(BigInteger,
                           nullable=False, index=True)

    # input
    input_set_series = Column(JSONB)

    # output
    output_set_series = Column(Float)
    
    
    __table_args__ = (
        UniqueConstraint('time_entry_ts', name='_time_entry_uc'),
    )





