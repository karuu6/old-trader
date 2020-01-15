def extract_features(save=False):
	prices   = []
	features = []
	files    = sorted(os.listdir(DATA_DIR), key=int)
	for fname in files:
		with open(DATA_DIR+fname, 'rb') as fp:
			f = []
			data = pickle.load(fp)

			t = data[0]
			b = data[1]
			n = 0
			for trade in t:
				if n == 0:
					prices.append(float(trade['price']))
					n+=1
				f.extend([float(trade['price']), float(trade['size']), 1. if trade
					['side'] == 'sell' else 0.])
			for entry in b['bids']:
				f.extend([j for j in map(float, entry)])	
			for entry in b['asks']:
				f.extend([j for j in map(float, entry)])
			features.append(f)
	if save:
		pickle.dump(features, open('features', 'wb'))
		pickle.dump(prices,   open('prices'  , 'wb'))
		return
	return np.array(features)/10000.0, np.array(prices)