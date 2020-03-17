"""
To run this script:
	python shoppingCost.py

If you run the above script, a correct calculateShoppingCost function should return:

The final cost for our shopping cart is 35.58
"""

import csv

def calculateShoppingCost(productPrices, shoppingCart):
	finalCost = 0
	"*** Add your code in here ***"
	for item in set(productPrices) & set(shoppingCart):
		finalCost += (float(productPrices.get(item)) * shoppingCart.get(item))  # (price * quantity)

	return finalCost


def createPricesDict(filename):
	productPrice = {}
	"*** Add your code in here ***"
	with open(filename, "r", encoding="utf-8") as inFile:
		reader = csv.reader(inFile)

		for row in reader:
			productPrice[row[0]] = row[1]  # key(1st column), value(2nd colum)

	return productPrice


if __name__ == '__main__':
	prices = createPricesDict("Grocery.csv")
	myCart = {"Bacon": 2,
		      "Homogenized milk": 1,
		      "Eggs": 5}
	print("The final cost for our shopping cart is {}".format(calculateShoppingCost(prices, myCart)))
