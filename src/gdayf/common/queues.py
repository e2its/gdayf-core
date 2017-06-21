## @package gdayf.common.queues
# Define all objects, functions and structured related to manage queues

from collections import deque
import heapq

## Class oriented to implement a queue
# abstraction for collections.deque over FIFO implementation
class Queue:
    ## Constructor
    def __init__(self):
        self.elements = deque()
    ## Method to check empty queue
    # @param self object pointer
    # @return 1 True 0 False
    def empty(self):
        return len(self.elements) == 0

    ## Method to insert element on queue
    # @param self object pointer
    # @param item element to be queued
    def put(self, item):
        self.elements.append(item)

    ## Method to get the first element
    # @param self object pointer
    # @param return element
    def get(self):
        return self.elements.popleft()

## Class oriented to implement a priorityqueue
# abstraction for collections.heapq
class PriorityQueue:
    ## Constructor
    def __init__(self):
        self.elements = []
    ## Method to check empty queue
    # @param self object pointer
    # @return 1 True 0 False
    def empty(self):
        return len(self.elements) == 0

    ## Method to insert element on queue
    # @param self object pointer
    # @param item element to be queued
    # @param priority numeric category
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    ## Method to get the most important element
    # @param self object pointer
    # @param return element
    def get(self):
        return heapq.heappop(self.elements)[1]

    ## Composite Method to insert a priorized element and get the most important element on same operation
    # @param self object pointer
    # @param item element to be queued
    # @param priority numeric category
    # @param return element
    def put_get(self, item, priority):
        return heapq.heappush(self.elements, (priority, item))[1]
