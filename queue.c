#include "huffman.h"
#include "queue.h"

/* Return the size of the queue. */
uint32_t queue_size(struct node *head) {
    uint32_t count = 0;
    struct node *iter;

    if (head) {
        iter = head;
        while (iter) {
            iter = iter->next;
            count++;
        }
    }

    return count;
}

/*
 * Add an element into the queue.
 *
 * The frequency of the node is taken into account
 * when adding a node into the queue. The priority
 * is in the order of increasing frequencies.
 */
void enqueue(struct node **head, struct node *new_node) {
    assert(*head);
    assert(new_node);

    struct node *temp, *prev;

    temp = *head;
    prev = NULL;

    while (temp) {
        if (new_node->data.freq < temp->data.freq)
            break;

        prev = temp;
        temp = temp->next;
    }

    if (!prev) {
        new_node->next = *head;
        *head = new_node;
        return;
    }

    new_node->next = prev->next;
    prev->next = new_node;
}

/*
 * Remove an element from the head of queue.
 *
 * Since the queue is a priority queue, the node
 * returned will have the least frequency.
 */
struct node *dequeue(struct node **head) {
    struct node *ret = NULL;

    if (*head) {
        ret = *head;
        *head = (*head)->next;
    }

    return ret;
}

/* Initialize a priority queue. */
void init_queue(struct node **head) {
    assert(*head);

    (*head)->next = NULL;
}

/*
 * Build a priority queue from a histogram.
 *
 * The character frequencies (value of the key) are used as
 * weights for ordering the queue.
 */
struct node *make_queue(struct map *fmap, uint32_t size) {
    uint32_t i;
    struct node *head = NULL, *temp = NULL;

    assert(fmap);

    for (i = 0; i < size; i++) {
        if (!head) {
            head = (struct node *) calloc(1, sizeof(struct node));
            assert(head);

            memcpy(&head->data, &fmap[i], sizeof(struct map));
            init_queue(&head);
        } else {
            temp = (struct node *) calloc(1, sizeof(struct node));
            assert(temp);

            memcpy(&temp->data, &fmap[i], sizeof(struct map));
            enqueue(&head, temp);
        }
    }

    return head;
}

/* Free all the elements in a queue. */
void nuke_queue(struct node **head) {
    struct node *nuke;

    while (*head)
        if ((nuke = dequeue(head)))
            free(nuke);
}
