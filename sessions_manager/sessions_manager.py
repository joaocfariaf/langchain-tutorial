import heapq
from typing import Any
from langchain_core.chat_history import InMemoryChatMessageHistory

class userSessionNumberManager:
    def __init__(self) -> None:
        self.session_count = 0
        self.session_heap = []

    def delete_session(self, session_number: int):
        heapq.heappush(self.session_heap, session_number)

    def new_session_number(self):
        if self.session_heap:
            return heapq.heappop(self.session_heap)
        else:
            self.session_count += 1
            return self.session_count


class sessionsManager:
    def __init__(self) -> None:
        self.id_manager = {}
        self.memo = {}

    def get_session_history(self, session_id: str)-> Any:
        try:
            return self.memo[session_id]
        except:
            raise KeyError('SESSION NOT FOUND')
    
    def delete_session(self, session_id: str):
        try:
            self.memo.pop(session_id)
            user_name, session_number = session_id.split('-')
            self.id_manager[user_name].delete_session(session_number)
        except:
            raise Warning('SESSION NOT FOUND')
     
    def create_session(
            self, 
            user_name: str
        ) -> str:
        if user_name not in self.id_manager:
            self.id_manager[user_name] = userSessionNumberManager()
        session_number = self.id_manager[user_name].new_session_number()
        session_id = f'{user_name}-{session_number}'
        self.memo[session_id] = InMemoryChatMessageHistory()

        return session_id

