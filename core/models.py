from sqlalchemy import (
    Column, Integer, BigInteger, Float, UniqueConstraint, ForeignKey, text, DateTime, String,  Text,
    JSON, Boolean, Numeric, Date, Index # Import the JSON type
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import JSONB


Base = declarative_base()

class Instrument(Base):
    """The Universal Registry. Unique by FIGI."""
    __tablename__ = "instruments"
    
    id = Column(Integer, primary_key=True, index=True)
    # The Anchor: Composite FIGI is the unique identifier for the Entity
    composite_figi = Column(String(20), unique=True, index=True, nullable=False) 
    
    ticker = Column(String(20)) # The *current* ticker (for reference only)
    name  = Column(String(255)) 
    market = Column(String(20)) 
    locale = Column(String(20)) 
    primary_exchange  = Column(String(20)) 
    type  = Column(String(20)) 
    active =  Column(Boolean, default=True) 
    currency_name = Column(String(20)) 
    cik = Column(BigInteger, nullable=True)
    share_class_figi = Column(String(20)) 

    
class UniverseMembership(Base):
    """Tracks the entry and exit of an instrument for a specific strategy/run."""
    __tablename__ = "universe_membership"
    
    id = Column(BigInteger, primary_key=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False)
    
    # This allows a stock to be added/dropped multiple times
    entry_date = Column(Date, nullable=False, index=True) 
    exit_date = Column(Date, nullable=True, index=True) 
    
    # Useful if you run multiple Monte Carlo seeds
    seed_id = Column(Integer, nullable=True, index=True)


class TickerMap(Base):
    """The Time-Series. Tracks the movement of tickers and the evolution of FIGIs."""
    __tablename__ = "ticker_map"
    
    id = Column(BigInteger, primary_key=True)
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False)
    
    # Ticker Lineage
    previous_ticker = Column(String(20), index=True) 
    ticker = Column(String(20), index=True) 
    previous_composite_figi = Column(String(20), nullable=True) 
    composite_figi = Column(String(20), index=True) 

    valid_from = Column(Date, nullable=False) 
    valid_to = Column(Date, nullable=True) 
    
    # Event Types: 'IPO', 'SYMBOL_CHANGE', 'ACQUISITION', 'SPIN_OFF', 'REUSE_GAP'
    change_event_type = Column(String(30))



class Dividend(Base):
    """Cash dividends linked to the specific Instrument Entity."""
    __tablename__ = "dividends"

    # Primary internal key
    id = Column(BigInteger, primary_key=True)
        # The Magic Link
    instrument_id = Column(Integer, ForeignKey('instruments.id'), nullable=False, index=True)

       # Polygon/Massive External ID for de-duplication (Upserts)
    external_id = Column(String(128), unique=True, index=True)
    
    # Snapshot
    ticker = Column(String(20)) 
    record_date  = Column(Date)
    pay_date = Column(Date)
    declaration_date = Column(Date)
    ex_dividend_date = Column(Date)
    frequency = Column(Integer)
    cash_amount = Column(Float)
    currency  = Column(String(20))
    distribution_type = Column(String(20))
    historical_adjustment_factor = Column(Float)
    split_adjusted_cash_amount = Column(Float)




class Quote(Base):
    __tablename__ = "quotes"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    instrument_id = Column(Integer, index=True, nullable=False)
    
    # Use BigInteger for nanoseconds since epoch
    t = Column(BigInteger, index=True, nullable=False) 

    # --- TRADE DATA ---
    o = Column(Numeric(18, 8))
    h = Column(Numeric(18, 8))
    l = Column(Numeric(18, 8))
    c = Column(Numeric(18, 8))
    v = Column(Numeric(24, 10), default=0) 
    vw = Column(Numeric(18, 8))

    # --- NBBO DATA ---
    bid = Column(Numeric(18, 8)) 
    ask = Column(Numeric(18, 8))
    bid_sz = Column(BigInteger) 
    ask_sz = Column(BigInteger)

    is_stale = Column(Boolean, default=False) 

    # Composite Index for lightning fast Lead-Lag lookups
    # __table_args__ = (
    #     Index('ix_instr_time', 'instrument_id', 't'),
    # )

    @property
    def mid_price(self):
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.c


# class OptionQuote(Base):
#     __tablename__ = "option_quotes"
#     id = Column(BigInteger, primary_key=True)
#     company_cik = Column(BigInteger, nullable=False) 
#     osi = Column(String(64), index=True, nullable=False)


#     # Boolean is more idiomatic
#     is_call = Column(Boolean, nullable=False)
#     contract_expiry = Column(DateTime, nullable=False, index=True)

#     t = Column(BigInteger, nullable=False, index=True) 
#     underlying_c = Column(Float) #  at t of the underlying asset close NEEDS LOOKUP. 
    
#     underlying_SOD = Column(Float) #  at t of the underlying asset close NEEDS LOOKUP. 
    
#     o   = Column(Float)     
#     h   = Column(Float)         
#     l   = Column(Float)         
#     c   = Column(Float)         
#     v   = Column(BigInteger)   
#     vw  = Column(Float)         
#     n   = Column(BigInteger)   
#     __table_args__ = (
#             UniqueConstraint('osi', 't', name='_option_minute_uc'),
#         )
    


# # https://www.alphavantage.co/query?function=FEDERAL_FUNDS_RATE&interval=daily&apikey=demo
# class FEDERAL_FUNDS_RATE(Base):
#     __tablename__ = "federal_funds_rate"
#     id = Column(BigInteger, primary_key=True)
#     t = Column(BigInteger, nullable=False, index=True)  # t  expressed in UNIX 
 
#     #daily follow the url, swap the demo key with the env key.
#     risk_free_rate_r = Column(Float) 





# class TREASURY_YIELD(Base):
#     __tablename__ = "treasury_yield"
#     id = Column(BigInteger, primary_key=True)
#     t = Column(BigInteger, nullable=False, index=True)  # t  expressed in UNIX 
#     _3month = Column(Float)

#     _2year =   Column(Float)
#     _5year =   Column(Float)
#     _7year =   Column(Float)
#     _10year =   Column(Float)



# # https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&time_from=20250901T0930&time_to=20251001T0930&limit=1000&apikey=JM19QBKIL3IFOLLT


# class NEWS_SENTIMENT(Base):
#     __tablename__ = "news_sentiment"
#     id = Column(BigInteger, primary_key=True)

#     title = Column(Text, nullable=False)
#     url = Column(String(1024), nullable=False, index=True) # URLs can be long
#     summary = Column(Text)
    

#     time_published_ts = Column(BigInteger, nullable=False, index=True) # convert     "time_published": "20251001T072600", into unix
#     source = Column(String(255))
#     source_domain = Column(String(255))
    
#     authors = Column(JSONB)
#     topics = Column(JSONB)
#     ticker_sentiment = Column(JSONB)
    
#     # Overall sentiment for the article
#     overall_sentiment_score = Column(Float)
#     overall_sentiment_label = Column(String(50))



# class Currency(Base):
#     __tablename__ = "currencies"
#     id = Column(Integer, primary_key=True)
#     code = Column(String(3), unique=True, index=True, nullable=False) 
#     name = Column(String(50))
#     individual_flow = Column(Float, default=0.0)
#     last_updated = Column(Date)

# class ForexQuote(Base):
#     __tablename__ = "forex_quotes"

#     id = Column(BigInteger, primary_key=True, autoincrement=True)

#     ticker = Column(String(50))
#     t = Column(BigInteger, nullable=False) 
#     o = Column(Float)
#     h = Column(Float)
#     l = Column(Float)
#     c = Column(Float)
#     n = Column(Float) 
#     v = Column(Float) 
#     vw = Column(Float)
#     __table_args__ = (
#             UniqueConstraint('ticker', 't', name='_forex_minute_uc'),
#         )

# # class Ratios(Base):
# #     __tablename__ = 'ratios'
    
# #     id = Column(BigInteger, primary_key=True)
# #     company_cik = Column(Integer, ForeignKey("companies.cik"), index=True, nullable=False)

# #     # --- UPDATED TEMPORAL COORDINATES (Relaxed for Ratios) ---
# #     # We remove nullable=False because Ratios are often just snapshots
# #     filing_date = Column(Date, index=True) 
# #     period_end  = Column(Date)          
# #     fiscal_year = Column(Integer)
# #     fiscal_quarter = Column(Integer)
# #     timeframe = Column(String(30)) 

# #     # --- MANDATORY SNAPSHOT DATE ---
# #     # This is the "Basis Vector" for Ratios
# #     date = Column(Date, nullable=False, index=True)

# #     # --- VALUATION MULTIPLES ---
# #     market_cap = Column(Numeric(20, 2))
# #     enterprise_value = Column(Numeric(20, 2))
# #     price_to_earnings = Column(Numeric(20, 4))
# #     price_to_sales = Column(Numeric(20, 4))
# #     price_to_book = Column(Numeric(20, 4))
# #     price_to_cash_flow = Column(Numeric(20, 4))
# #     price_to_free_cash_flow = Column(Numeric(20, 4))
# #     ev_to_sales = Column(Numeric(20, 4))
# #     ev_to_ebitda = Column(Numeric(20, 4))

# #     # --- PROFITABILITY & RETURNS ---
# #     earnings_per_share = Column(Numeric(20, 4))
# #     return_on_assets = Column(Numeric(20, 4))
# #     return_on_equity = Column(Numeric(20, 4))
# #     dividend_yield = Column(Numeric(20, 4))

# #     # --- LIQUIDITY & SOLVENCY ---
# #     current = Column(Numeric(20, 4)) # Current Ratio
# #     quick = Column(Numeric(20, 4))   # Quick Ratio
# #     debt_to_equity = Column(Numeric(20, 4))
# #     cash = Column(Numeric(20, 2))
# #     free_cash_flow = Column(Numeric(20, 2))

# #     # --- MARKET / VOLUME DATA ---
# #     price = Column(Numeric(20, 2))
# #     average_volume = Column(BigInteger)
# #     date = Column(Date) # Specific price date used for ratio calculation

# #     # --- METADATA ---
# #     cik = Column(String(20))
# #     tickers = Column(JSONB)
# #     additional_data = Column(JSONB, server_default='{}')




# class ShortInterest(Base):
#     __tablename__ = 'short_interest'
    
#     id = Column(BigInteger, primary_key=True)
    
#     # Foreign Key Linking
#     company_cik = Column(Integer, ForeignKey("companies.cik"), index=True, nullable=False)
    
#     # --- DATA FROM API RESPONSE ---
#     # "settlement_date": "2025-03-14"
#     settlement_date = Column(Date, nullable=False, index=True)
    
#     # "short_interest": 3906231
#     short_interest = Column(BigInteger)
    
#     # "avg_daily_volume": 2340158
#     avg_daily_volume = Column(BigInteger)
    
#     # "days_to_cover": 1.67
#     days_to_cover = Column(Float)
    
#     # "ticker": "A" (Redundant but useful for verification)
#     ticker = Column(String(10))

#     __table_args__ = (
#         UniqueConstraint('company_cik', 'settlement_date', name='uq_short_interest_identity'),
#     )






# class CashFlow(Base):
#     __tablename__ = 'cash_flow'
    
#     id = Column(BigInteger, primary_key=True)
#     company_cik = Column(Integer, ForeignKey("companies.cik"), index=True, nullable=False)

#     # --- TEMPORAL COORDINATES ---
#     filing_date = Column(Date, nullable=False, index=True) 
#     period_end  = Column(Date, nullable=False)          
#     fiscal_year = Column(Integer, nullable=False)
#     fiscal_quarter = Column(Integer)
#     timeframe = Column(String(30)) 

#     # --- OPERATING ACTIVITIES ---
#     net_income = Column(Numeric(20, 2))
#     depreciation_depletion_and_amortization = Column(Numeric(20, 2))
#     change_in_other_operating_assets_and_liabilities_net = Column(Numeric(20, 2))
#     income_loss_from_discontinued_operations = Column(Numeric(20, 2))
#     net_cash_from_operating_activities = Column(Numeric(20, 2))
#     cash_from_operating_activities_continuing_operations = Column(Numeric(20, 2))
#     net_cash_from_operating_activities_discontinued_operations = Column(Numeric(20, 2))
#     other_operating_activities = Column(Numeric(20, 2))

#     # --- INVESTING ACTIVITIES ---
#     purchase_of_property_plant_and_equipment = Column(Numeric(20, 2))
#     sale_of_property_plant_and_equipment = Column(Numeric(20, 2))
#     net_cash_from_investing_activities = Column(Numeric(20, 2))
#     net_cash_from_investing_activities_continuing_operations = Column(Numeric(20, 2))
#     net_cash_from_investing_activities_discontinued_operations = Column(Numeric(20, 2))
#     other_investing_activities = Column(Numeric(20, 2))

#     # --- FINANCING ACTIVITIES ---
#     dividends = Column(Numeric(20, 2))
#     short_term_debt_issuances_repayments = Column(Numeric(20, 2))
#     long_term_debt_issuances_repayments = Column(Numeric(20, 2))
#     net_cash_from_financing_activities = Column(Numeric(20, 2))
#     net_cash_from_financing_activities_continuing_operations = Column(Numeric(20, 2))
#     net_cash_from_financing_activities_discontinued_operations = Column(Numeric(20, 2))
#     noncontrolling_interests = Column(Numeric(20, 2))
#     other_financing_activities = Column(Numeric(20, 2))

#     # --- ADJUSTMENTS & RECONCILIATION ---
#     effect_of_currency_exchange_rate = Column(Numeric(20, 2))
#     change_in_cash_and_equivalents = Column(Numeric(20, 2))
#     other_cash_adjustments = Column(Numeric(20, 2))

#     # --- METADATA ---
#     cik = Column(String(20))
#     tickers = Column(JSONB)
#     additional_data = Column(JSONB, server_default='{}')


#     __table_args__ = (
#         UniqueConstraint('company_cik', 'filing_date', 'period_end', 'timeframe', name='uq_cash_flow_identity'),
#     )





# class BalanceSheet(Base):
#     __tablename__ = 'balance_sheets'
    
#     id = Column(BigInteger, primary_key=True)

#     company_cik = Column(Integer, ForeignKey("companies.cik"), index=True, nullable=False)

#     # --- TEMPORAL COORDINATES ---
#     filing_date = Column(Date, nullable=False, index=True) 
#     period_end  = Column(Date, nullable=False)          
#     fiscal_year = Column(Integer, nullable=False)
#     fiscal_quarter = Column(Integer)
#     timeframe = Column(String(30)) 

#     # --- ASSETS ---
#     cash_and_equivalents = Column(Numeric(20, 2))
#     short_term_investments = Column(Numeric(20, 2))
#     receivables = Column(Numeric(20, 2))
#     inventories = Column(Numeric(20, 2))
#     other_current_assets = Column(Numeric(20, 2))
#     total_current_assets = Column(Numeric(20, 2))
    
#     property_plant_equipment_net = Column(Numeric(20, 2))
#     goodwill = Column(Numeric(20, 2))
#     intangible_assets_net = Column(Numeric(20, 2))
#     other_assets = Column(Numeric(20, 2))
#     total_assets = Column(Numeric(20, 2))

#     # --- LIABILITIES ---
#     accounts_payable = Column(Numeric(20, 2))
#     debt_current = Column(Numeric(20, 2))
#     deferred_revenue_current = Column(Numeric(20, 2))
#     accrued_and_other_current_liabilities = Column(Numeric(20, 2))
#     total_current_liabilities = Column(Numeric(20, 2))
    
#     long_term_debt_and_capital_lease_obligations = Column(Numeric(20, 2))
#     other_noncurrent_liabilities = Column(Numeric(20, 2))
#     total_liabilities = Column(Numeric(20, 2))
#     commitments_and_contingencies = Column(Numeric(20, 2))

#     # --- EQUITY ---
#     preferred_stock = Column(Numeric(20, 2))
#     common_stock = Column(Numeric(20, 2))
#     additional_paid_in_capital = Column(Numeric(20, 2))
#     retained_earnings_deficit = Column(Numeric(20, 2))
#     accumulated_other_comprehensive_income = Column(Numeric(20, 2))
#     treasury_stock = Column(Numeric(20, 2))
#     other_equity = Column(Numeric(20, 2))
    
#     total_equity_attributable_to_parent = Column(Numeric(20, 2))
#     noncontrolling_interest = Column(Numeric(20, 2))
#     total_equity = Column(Numeric(20, 2))
#     total_liabilities_and_equity = Column(Numeric(20, 2))

#     # --- METADATA ---
#     cik = Column(String(20))
#     tickers = Column(JSONB)
#     additional_data = Column(JSONB, server_default='{}')
#     __table_args__ = (
#         UniqueConstraint('company_cik', 'filing_date', 'period_end', 'timeframe', name='uq_balance_sheets_identity'),
#     )






# class IncomeStatement(Base):
#     """
#     Standardized Financial Manifold: Income Statement
#     Basis Vectors defined by 5% Global Observation Density Sieve.
#     """
#     __tablename__ = 'income_statements'
        
#     id = Column(BigInteger, primary_key=True)

#     company_cik = Column(Integer, ForeignKey("companies.cik"), index=True, nullable=False)
    
#     # --- TEMPORAL COORDINATES ---
#     # We use BigInteger for Unix timestamps to avoid timezone friction
#     filing_date = Column(Date, nullable=False, index=True) 
#     period_end  = Column(Date, nullable=False)         
#     fiscal_year = Column(Integer, nullable=False)
#     fiscal_quarter = Column(Integer)
#     timeframe = Column(String(30))
#     # --- THE WHITELIST BASIS VECTORS (NUMERIC) ---
#     # Using Numeric(20, 2) for exact decimal precision (Conservation of Mass)
    
#     # Revenue & Flow
#     revenue = Column(Numeric(20, 2))
#     cost_of_revenue = Column(Numeric(20, 2))
#     gross_profit = Column(Numeric(20, 2))
    
#     # Operating Friction
#     operating_income = Column(Numeric(20, 2))
#     total_operating_expenses = Column(Numeric(20, 2))
#     selling_general_administrative = Column(Numeric(20, 2))
#     research_development = Column(Numeric(20, 2))
#     other_operating_expenses = Column(Numeric(20, 2))
#     depreciation_depletion_amortization = Column(Numeric(20, 2))
    
#     # Non-Operating & Stability Keys
#     ebitda = Column(Numeric(20, 2)) # Management-reported proxy
#     interest_income = Column(Numeric(20, 2))
#     interest_expense = Column(Numeric(20, 2))
#     total_other_income_expense = Column(Numeric(20, 2))
#     other_income_expense = Column(Numeric(20, 2))
#     equity_in_affiliates = Column(Numeric(20, 2))
    
#     # The Bottom Line (Ground Truth)
#     income_before_income_taxes = Column(Numeric(20, 2))
#     income_taxes = Column(Numeric(20, 2))
#     consolidated_net_income_loss = Column(Numeric(20, 2))
#     net_income_loss_attributable_common_shareholders = Column(Numeric(20, 2))
#     noncontrolling_interest = Column(Numeric(20, 2))
#     preferred_stock_dividends_declared = Column(Numeric(20, 2))
#     discontinued_operations = Column(Numeric(20, 2))
    
#     # Per-Share/Equity Dimensions
#     basic_shares_outstanding = Column(Numeric(20, 2))
#     diluted_shares_outstanding = Column(Numeric(20, 2))
#     basic_earnings_per_share = Column(Numeric(20, 5))
#     diluted_earnings_per_share = Column(Numeric(20, 5))
    
#     # Metadata / Identifiers
#     cik = Column(String(20))
#     tickers = Column(JSONB) # Store list of tickers if re-used
    
#     # --- THE RESIDUAL BLOB ---
#     # Anything failing the 5% sieve drops into this 'Blacklist' bin
#     additional_data = Column(JSONB, server_default='{}')

#     __table_args__ = (
#         UniqueConstraint('company_cik', 'filing_date', 'period_end', 'timeframe', name='uq_income_statements_identity'),
#     )




