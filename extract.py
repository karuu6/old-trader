from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import numpy as np
from conf import *
import pickle
import random
import json
import sys
import os
import gc

def extract_features(save=False):
	features = []
	prices = []
	files  = sorted(os.listdir(DATA_DIR), key=int)
	for fname in files:
		with open(DATA_DIR+fname, 'rb') as fp:
			feats = []

			data = pickle.load(fp)
			book = data[1]
			trades = data[0]

			mp   = (float(book['bids'][0][0])+float(book['asks'][0][0]))/2
			s_bid = sum([float(i[1]) for i in book['bids']])
			n_bid = sum([float(i[2]) for i in book['bids']])
			s_ask = sum([float(i[1]) for i in book['asks']])
			n_ask = sum([float(i[2]) for i in book['asks']])

			a_size = (s_bid + s_ask) / 2

			div_bid = np.array([mp, s_bid, n_bid])
			div_ask = np.array([mp, s_ask, n_ask])

			for i in book['bids']:
				feats.extend(np.array([i for i in map(float,i)])/div_bid)
			for i in book['asks']:
				feats.extend(np.array([i for i in map(float,i)])/div_ask)
			for i in trades:
				feats.extend([float(i['price'])/mp, float(i['size'])/a_size, 1 if i['side'] == 'sell' else 0])


			features.append(feats)
			prices.append(mp)
	
	gc.collect()

	if save:
		json.dump(features, open('features', 'w'))
		json.dump(prices,   open('prices',   'w'))
		return

	return np.array(features),np.array(prices)

def plot_prices(prices, extrema=([],[]), fmt='-'):
	x = np.linspace(0, len(prices)-1, len(prices))
	
	plt.scatter(extrema[0], extrema[1], c='r')
	plt.scatter(extrema[2], extrema[3], c='g')
	plt.plot(x, prices, fmt)
	plt.show()

def label_samples(features, prices, order=600, n=1):
	x = np.linspace(0, len(prices)-1, len(prices))
	sells = argrelextrema(prices, np.greater_equal, order=order)[0]
	buys  = argrelextrema(prices, np.less_equal,    order=order)[0]
	
	sells = extend_extrema(sells, len(prices)-1, n=n)
	buys  = extend_extrema(buys,  len(prices)-1, n=n)


	br  = (len(sells) + len(buys))/2
	fix = []
	while len(fix) < br:
		k = random.choice(x)
		if k not in sells and k not in buys:
			fix.append(k)

	y=np.zeros(int(br*2)+len(fix))
	feats=[]
	
	n=0
	for i in fix:
		feats.append(features[int(i)])
		y[n] = HOLD
		n+=1

	for i in buys:
		feats.append(features[int(i)])
		y[n] = BUY
		n+=1

	for i in sells:
		feats.append(features[int(i)])
		y[n] = SELL
		n+=1

	return np.array(feats), y

def extend_extrema(x, m, n=20):
	l=[]
	for i in x:
		if i+n > m:
			l.extend([j for j in range(i-n,i+1)])
		elif i-n < 0:
			l.extend([j for j in range(i,i+1+n)])
		else:
			l.extend([j for j in range(i-n,i+n+1)])
	return l


if __name__ == '__main__':
	#prices = None
	with open('s_data/1_sec_prices', 'r') as fp:
		prices = np.array(json.load(fp))

	#_, prices = extract_features()

	b  = argrelextrema(prices, np.less_equal,    order=600)[0]
	b  = extend_extrema(b, len(prices), n=5)
	by = [prices[i] for i in b]
	s  = argrelextrema(prices, np.greater_equal, order=600)[0]
	s  = extend_extrema(s, len(prices), n=5)
	sy = [prices[i] for i in s]
	

	plot_prices(prices, (b, by, s, sy))