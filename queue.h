#ifndef __QUEUE_H__
#define __QUEUE_H__

/* Return the size of the queue. */
uint32_t queue_size(struct node *head);
void enqueue(struct node **head, struct node *new_node);
struct node *dequeue(struct node **head);
void init_queue(struct node **head);
struct node *make_queue(struct map *fmap, uint32_t size);
void nuke_queue(struct node **head);

#endif /* DEFINE __QUEUE_H__ */


