import pickle

class Pickler():
    def save(obj, loc):
        try:
            with open(loc, 'wb') as out:
                pickle.dump(obj, out, pickle.HIGHEST_PROTOCOL)
            return(True)
        except:
            return(False)

    def load(loc):
        try:
            with open(loc, 'rb') as inp:
                return(pickle.load(inp))
        except:
            return(False)