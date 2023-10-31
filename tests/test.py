import queue

q = queue.Queue(maxsize=10)

q.put(1)
q.put(2)
q.put(3)
q.queue.clear()
q.put(4)

print(q.queue)