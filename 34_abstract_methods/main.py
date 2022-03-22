from abc import ABC, abstractmethod

class TrataDados(ABC):

    def __init__(self, dados:list):
        self.dados = dados

    @property
    def get_dados(self):
        return self.dados

    @abstractmethod
    def score(self):
        pass


class CalculaScore(TrataDados):

    def score(self):
        return sum(self.dados)

class FinalScore(TrataDados):
    
    def score(self):
        return [round(item/sum(self.dados),2) for item in self.dados]


if __name__ == '__main__':
    DADOS = [1,2,3]

    calcula_score = CalculaScore(DADOS)

    print(calcula_score.get_dados)
    print(calcula_score.score())

    final_score = FinalScore(DADOS)

    print(final_score.get_dados)
    print(final_score.score())


