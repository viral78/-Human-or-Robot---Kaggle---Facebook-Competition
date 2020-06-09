import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
import xgboost
from sklearn.ensemble import RandomForestClassifier
from google.colab import files
  
bids = pd.read_csv('bids.csv') 
train = pd.read_csv('train.csv')  
test = pd.read_csv('test.csv')

# total number of bids feature

train['total_bids'] = 0
for bidder in train['bidder_id']:
    bid_count = bids[bids.bidder_id == bidder].count()[0]
    train.loc[train[train.bidder_id == bidder].index, 'total_bids'] = bid_count


test['total_bids'] = 0
for bidder in test['bidder_id']:
    bid_count = bids[bids.bidder_id == bidder].count()[0]
    test.loc[test[test.bidder_id == bidder].index, 'total_bids'] = bid_count


# total number of unique auctions feature

train['total_auctions'] = 0
for bidder in train['bidder_id']:
    auction_count = bids[bids.bidder_id == bidder]['auction'].nunique()
    train.loc[train[train.bidder_id == bidder].index, 'total_auctions'] = auction_count

test['total_auctions'] = 0
for bidder in test['bidder_id']:
    auction_count = bids[bids.bidder_id == bidder]['auction'].nunique()
    test.loc[test[test.bidder_id == bidder].index, 'total_auctions'] = auction_count

# total number of unique devices feature

train['total_devices'] = 0
for bidder in train['bidder_id']:
    device_count = bids[bids.bidder_id == bidder]['device'].nunique()
    train.loc[train[train.bidder_id == bidder].index, 'total_devices'] = device_count

test['total_devices'] = 0
for bidder in test['bidder_id']:
	device_count = bids[bids.bidder_id == bidder]['device'].nunique()
    test.loc[test[test.bidder_id == bidder].index, 'total_devices'] = device_count

# total number of unique ips feature

train['no_of_ips'] = 0
for bidder in train['bidder_id']:
    ip_count = bids[bids.bidder_id == bidder]['ip'].nunique()
    train.loc[train[train.bidder_id == bidder].index, 'no_of_ips'] = ip_count

test['no_of_ips'] = 0
for bidder in test['bidder_id']:
    ip_count = bids[bids.bidder_id == bidder]['ip'].nunique()
    test.loc[test[test.bidder_id == bidder].index, 'no_of_ips'] = ip_count

train.to_csv('train_features.csv', sep=',',index=False)
test.to_csv('test_features.csv', sep=',',index=False)