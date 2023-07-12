from tenacity import retry
import numpy as np 

@retry
def simple_retry():
	print('run')
	x = np.random.randint(1,5)
	desired = 3
	if x != desired:
		raise(f'Number generated was not {desired}')


if __name__ == '__main__':
	simple_retry()
