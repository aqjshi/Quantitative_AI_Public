from fredapi import Fred
fred = Fred(api_key='6799bbe3b5be9dc92751b055763e402b')

# Pulling the components for the 1-X-N theory
u3_rate = fred.get_series('UNRATE') # Official X
u6_rate = fred.get_series('U6RATE') # Official underemployment
part_time_econ = fred.get_series('LNS12032194') # Part of the "N" driver pool
self_employed = fred.get_series('LNS12027714') # Gig-worker proxy

# Calculate the 'Underemployment Delta'
underemployment_delta = u6_rate - u3_rate
print((underemployment_delta).mean())