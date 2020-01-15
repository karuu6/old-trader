from data import Req
import numpy as np
from conf import *
import pickle
import gc


class Market(Req):
	def __init__(self, symbol, initial_qty, model):
		super().__init__(symbol)
		self.base  = 0.0
		self.quote = initial_qty
		self.model = model
		self.state = 'BUY'

		self.lp = 0.0

	def save(self, n):
		if n%30 == 0:
			print('Quote balance: %f'   % (self.quote))
			print('Base  balance: %f\n' % (self.base))		

		cur_p = (float(self.book['bids'][0][0])+float(self.book['asks'][0][0]))/2

		feats = extract_features((self.trades,self.book))
		p = self.model.predict(feats)[0]

		if p == HOLD:
			return
		elif p == BUY:
			if self.state == 'BUY':
				self.base  = execute_order(self.book, self.quote, 'buy')
				#self.base  = self.quote/cur_p
				self.quote = 0.0
				self.state = 'SELL'
				self.lp = cur_p
			return
		else:
			if self.state == 'SELL' and cur_p > self.lp:
			#if self.state == 'SELL':
				self.quote = execute_order(self.book, self.base, 'sell')
				#self.quote = cur_p*self.base
				self.base  = 0.0
				self.state = 'BUY'
				self.lp = cur_p
			return

def execute_order(book, amt, side):
	s = 0
	if side == 'sell':
		for i in book['bids']:
			i = [j for j in map(float, i)]
			if i[1] >= amt:
				s += i[0]*amt
				break
			 
			s   += i[0]*i[1]
			amt =  amt - i[1]
		return 0.997*s
	else:
		for i in book['bids']:
			i = [j for j in map(float, i)]
			qty = amt / i[0]
			if i[1] >= qty:
				s += qty
				break

			amt -= i[1]*i[0]
			s   += i[1]
		return 0.997*s

def extract_features(data):
	features = []
	feats = []

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
	return features


if __name__ == '__main__':
	ml = pickle.load(open('NOG', 'rb'))
	#ml = ML()
	m = Market('BTC-USD', 100.0, ml)
	m.start()