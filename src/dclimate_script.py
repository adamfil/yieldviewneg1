import requests 
import json 

my_token = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjI1MzQwMjMwMDc5OSwiaWF0IjoxNjMwNTMwMTgwLCJzdWIiOiJhMDIzYjUwYi0wOGQ2LTQwY2QtODNiMS1iMTExZDA2Mzk1MmEifQ.qHy4B0GK22CkYOTO8gsxh0YzE8oLMMa6My8TvhwhxMk'

# Fetch state codes
my_url = 'https://api.dclimate.net/apiv4/rma-code-lookups/valid_states'
head = {"Authorization": my_token}
r = requests.get(my_url, headers=head)
state_codes = r.json()

with open('state_codes.json', 'w') as file:
    json.dump(state_codes, file)

# Fetch counties for each state
my_url = 'https://api.dclimate.net/apiv4/rma-code-lookups/valid_counties/'
state_county = {}
for key in state_codes.keys():
    r = requests.get(my_url + str(key), headers=head)
    county_codes = r.json()
    state_county[key] = county_codes

with open('state_county.json', 'w') as file:
    json.dump(state_county, file)
