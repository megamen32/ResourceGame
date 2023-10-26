# Определение класса ресурса (яблока)
class Resource:
    def __init__(self, x, y,amount=1):
        self.x = x
        self.y = y
        self.amount = amount  # Количество ресурса в яблоке
    def get_visible_state(self):
        return [self.x,self.y,self.amount]
