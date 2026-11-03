import csv
import json


# ends with Total rows: 114 
# decrease from Total rows: 499 unfiltered





# Your exact raw text block wrapped into a clean variable
raw_blacklist_data = {
        "International Data":	32263,
        "U.S. Regional Data":	3008,
        "Academic Data":	33060,
        "Daily Rates":	94,	
        "Monthly Rates":	95,	
        "Annual Rates":	32219,
        "By Country":	158, 
        "Consumer Financial Conditions":	33119,
        "Stock Market Indexes":	32255,
        "Patents":	33959,
        "Cryptocurrencies":	33913,
        "Health Care Indexes":	33717	,
        "House Price Indexes":	32261	,
        "Housing Units Authorized, But Not Yet Started":	32301,
        "Housing Units Under Construction":	32303	,
        "Housing Units Completed":	32304	,
        "Manufactured Housing"	:33949	, 
        "Foreign Exchange Intervention":	32145,
        "AMERIBOR Benchmark Rates":	34009	,
        "Automobile Loan Rates":	33058	,
        "Bankers Acceptance Rate":	51	,
        "Certificates of Deposit":	121	,
        "Corporate Bonds":	32348	,
        "Credit Card Loan Rates":	33059	,
        "EONIA Rates":	34005	,
        "Euro Short-Term Rate":	34007	,
        "Eurodollar Deposits":	32298	,
        "Interest Checking Accounts":	33056	,
        "Interest Rate Swaps":	32299	,
        "Long-Term Securities":	32995	,
        "Monetary Policy":	34242	,
        "Money Market Accounts":	33055	,
        "Personal Loan Rates":	33057	,
        "Prime Bank Loan Rate":	117	,
        "Saving Accounts":	33491	,
        "SONIA Rates":	34003	,
        "Treasury Bills":	116	,
        "Treasury Inflation-Indexed Securities":	82	, 


        "Reserves":	123	,
        "M1 and Components":	25	,

        "M2 Minus Small Time Deposits":	96	,
        "M3 and Components":	28	,
        "MZM":	30	,
        "Memorandum Items":	26	,
        "Money Velocity":	32242	,
        "Borrowings":	122	,
        "Factors Affecting Reserve Balances":	32215	,
        "Securities, Loans, & Other Assets & Liabilities Held by Fed":	32218	,

        "Banking Indexes":	34078	,

        "Condition of Banks":	83	,
        "Consumer Credit":	101	,
        "Delinquencies and Delinquency Rates":	32440	,
        "Failures and Assistance Transactions"	:33121	,
        "8th District Banking Performance":	64	,
        "Mortgage Debt Outstanding":	33445	,
        "Net Charge-Offs and Charge-Off Rates":	32439	,
        "Securities & Investments":	99	,
        "Senior Credit Officer Opinion Survey":	34111	, 

        "Commercial and Industrial Loans by Time that Pricing Terms Were Set and by Commitment":	32406	,
        "Commercial and Industrial Loans Backed by Small Business Association":	33439	,

        "Commercial and Industrial Loans Made by Domestic Banks":	32370	,
        "Commercial and Industrial Loans Made by Large Domestic Banks":	32379	,
        "Commercial and Industrial Loans Made by Small Domestic Banks":	32388	,
        "Commercial and Industrial Loans Made by U.S. Branches and Agencies of Foreign Banks":	32397	,
        "Commercial and Industrial Loans Made Under Participation or Syndication":	33440	,
        "Civilian Labor Force"	:32442	,
        "Employment":	32444	,
        "Employment Population Ratio":	32445	,
        "Not in Labor Force":	32448	,
        "Duration of Unemployment":	32451	,
        "Losers and Leavers":	32452	,
        "Earnings":	33501	,
        "Entrants and Reentrants":	32453	,
        "Labor Force Status Flows":	33502	,



        "Total Private":	32306	,
        "Goods-Producing":	32307	,
        "Service-Providing"	:32326	,
        "Private Service-Providing"	:32308	,
        "Mining and Logging"	:32309	,
        "Construction"	:32310	,
        "Manufacturing"	:32311	,
        "Durable Goods":	32312	,
        "Nondurable Goods"	:32313	,
        "Trade, Transportation, and Utilities"	:32314	,
        "Wholesale Trade":	32315	,
        "Retail Trade"	:32316	,
        "Transportation and Warehousing":	32317	,
        "Utilities"	:32318	,
        "Information":	32319	,
        "Financial Activities"	:32320	,
        "Professional and Business Services":	32321	,
        "Education and Health Services":	32322	,
        "Leisure and Hospitality":	32323	,
        "Other Services":	32324	,
        "Government":	32325	,

  
        "Hires (Levels and Rates)": 32245	,
        "Total Separations (Levels and Rates)":	32246	,
        "Layoffs and Discharges (Levels and Rates)":	32248	,
        "Other Separations (Levels and Rates)":	32249	,


        "Poverty Measures":	33735,
        "Supplemental Nutrition Assistance Program":	33514,
        
        
        "Productivity & Costs": 2,


        "Domestic Capital Account (Saving & Investment)":	112	,
        "Foreign Transactions":	108	,
        "Fixed Assets":	33697	,
        "Gross Domestic Income":	33122	,
        "GDP/GNP":	106	,
        "Gov't Receipts, Expenditures & Investment":	107	,
        "Health Care Spending":	33719	,
        "Imputations":	33054	,
        "Industry"	:33045	,
        "Price Indexes & Deflators"	:21	,
        "Private Enterprise Income"	:109	,
        "Quantity Indexes"	:33021	,
        "Effect of ARRA on Selected NIPA Estimates":	33401	, 


        "Flow of Funds": 	32251,

        "Exports":	16	,
        "Imports":	17	,
        "Income Payments & Receipts":	3000	,
        "International Investment Position":	33705	,
        "U.S. International Finance":	127	,

        "Trade Indexes"	: 32220,




        "Inventories":32432	,

        "Unfilled Orders":	32435	,
        "Unfilled Orders to Shipments":	32434	,
        "Shipments"	:32430	, 


        "Food and Beverages":	32415	,
        "Apparel"	:32417	,
        "Transportation"	:32418	,
        "Medical Care":	32419	,
        "Recreation"	:32420	,
        "Education and Communication":	32421	,
        "Other Goods and Services"	:32422	,
        "Special Indexes"	:32424	, 
      



        "Domestically Chartered Commercial Banks":	33079	,
        "Foreign-Related Institutions":	33080	,


        
        "All Commercial and Industrial Loans":	32362	,
        "Base Rate of Loans":	32369	,
        "Daily Repricing/Maturity Interval":	32364	,
        "More Than 365 Days Repricing/Maturity Interval":	32367	,
        "Size of Loans"	:32368	,
        "31 to 365 Days Repricing/Maturity Interval":	32366	,
        "2 to 30 Days Repricing/Maturity Interval"	:32365	,
        "Zero Repricing/Maturity Interval"	:32363	,

        "Employer Contributions":	33031	,
        "Personal Current Taxes":	33032	,
        "Wage and Salary Accruals":	33030	,

        "Accommodation Services":	33564	,
        "Advertising Space and Time Sales":	33548	,
        "Cleaning and Building Maintenance Services":	33560	,
        "Construction Commodity Based":	33573	,
        "Contract Work on Textile Products, Apparel, and Leather":	33572	,
        "Credit Intermediation Services":33551	,
        "Data Processing and Related Services"	:33550	,
        "Durability of Product":	33574	,
        "Educational Services":	33563	,
        "Employment Services"	:33557	,
        "Entertainment Services"	:33567	,
        "Farm Products"	:33528	,
        "Final Demand"	:33575	,
        "Food and Beverage for Immediate Consumption Services"	:33565	,

        "Furniture and Household Durables"	:33539	,
        "Health Care Services"	:33561	,
        "Hides, Skins, Leather, and Related Products"	:33531	,
        "Inputs to Industries"	:33582	,
        "Insurance and Annuities"	:33553	,
        "Intermediate Demand By Commodity Type":	33577	,
        "Intermediate Demand By Production Flow"	:33576	,
        "Investment Services"	:33552	,
        "Metal Treatment Services"	:33570	,
        "Mining Services"	:33571	,
        "Miscellaneous Products"	:33542	,
        "Network Compensation from Broadcast and Cable Television and Radio"	:33547	,
        "Nonmetallic Mineral Products"	:33540	,
        "Processed Foods and Feeds"	:33529	,
        "Professional Services"	:33556	,
        "Publishing Sales, Excluding Software"	:33545	,
        "Pulp, Paper, and Allied Products"	:33536	,
        "Real Estate Services"	:33554	,
        "Rental and Leasing of Goods"	:33555	,
        "Repair and Maintenance Services"	:33566	,
        "Retail Trade Services"	:33569	,
        "Selected Security Services"	:33559	,
        "Software Publishing"	:33546	,
        "Special Indexes Commodity Based"	:33580	,
        "Stage of Processing"	:33581	,
        "Telecommunication, Cable, and Internet User Services"	:33549	,
        "Transportation Equipment"	:33541	,
        "Transportation Services"	:33543	,
        "Travel Arrangement Services"	:33558	,
        "Warehousing, Storage, and Related Services"	:33544	,
        "Waste Collection and Remediation Services"	:33562	,
        "Wholesale Trade Services"	:33568	

    }

output_file = "upsert_categories_blacklist.csv"

# Writing out to CSV with clean headers
with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    
    # Write column names
    writer.writerow(["category_name", "category_id"])
    
    # Write sorted key-value rows for structured layout
    for category_name, category_id in sorted(raw_blacklist_data.items()):
        writer.writerow([category_name.strip(), category_id])
        print(f"{category_id},")


print(f"File successfully created: '{output_file}' containing {len(raw_blacklist_data)} rows.")