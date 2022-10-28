from datetime import datetime as dt

from dateutil.relativedelta import relativedelta

def shift_month(raw, shift):
    if len(raw) == 8:
        _format = '%Y%m%d'
        date = dt.strptime(str(raw),_format) + relativedelta(months = shift) 
        date_str = date.strftime(_format)
    if len(raw) == 6:
        _format = '%Y%m'
        date = dt.strptime(str(raw),_format) + relativedelta(months = shift) 
        date_str = date.strftime(_format)
    return date_str


if __name__ == '__main__':
	a = 20210401
	b = 202104

	a = str(a)
	shifted_a = shift_month(a,-1)
	print(shifted_a)

	b = str(b)
	shifted_b = shift_month(b,-1)
	print(shifted_b)
