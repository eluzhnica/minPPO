from transformers import AutoTokenizer, GPT2Tokenizer
from typing import Tuple
from gpt.bpe import BPETokenizer
from typing import Callable, List
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from gpt.gpt import GPT
import copy
from datasets import load_dataset

from transformers import pipeline
from transformers import AutoTokenizer, GPT2Tokenizer
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class LMEnv:
    reward_func: Callable[[list[str]], float]
    max_length: int
    model: GPT
    base_model: GPT
    tokenizer: GPT2Tokenizer

    def __init__(self, model: GPT, base_model: GPT, tokenizer: GPT2Tokenizer, max_length: int,
                 reward_func: Callable[[list[str]], float], prompt_dataset: list[str], stop_words: list[str],
                 device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reward_func = reward_func
        self.base_model = base_model
        self.prompt_dataset = prompt_dataset
        self.device = device
        self.stop_words = stop_words

    def reset(self) -> str:
        """
        Reset the environment and return the initial state.
        """
        # get a random prompt from the prompt dataset
        # encode the prompt
        # return the encoded prompt
        prompt = random.choice(self.prompt_dataset)
        # prompt = self.tokenizer.encode(prompt)
        # return torch.LongTensor(prompt, device=self.device)
        return prompt

    def calculate_reward(self, idx: torch.LongTensor, logits: torch.FloatTensor) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        """
        Calculate the reward for a given sequence of indices. This is done by decoding the
        indices into a string and then passing that string to the reward function.

        idx: LongTensor of shape (b, t)
        logits: FloatTensor of shape (b, t, vocab_size)
        """
        # get the rewards
        # calculate the KL divergence between the base model and the model
        # merge the rewards and the divergence into a single reward
        # return the reward

        reward = self.reward_func(self.tokenizer.batch_decode(idx))  # (b,)
        reward = torch.tensor(reward, device=self.device)  # (b,)
        # calculate the logits of the base model
        base_logits, _, _ = self.base_model(idx)  # (b, t, vocab_size)
        # calculate the KL divergence between the base model and the model in every token
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # get log probs only for the tokens that were generated
        base_log_probs = torch.gather(base_log_probs, 2, idx.unsqueeze(2)).squeeze(-1)  # (b, t) after squeeze
        log_probs = torch.gather(log_probs, 2, idx.unsqueeze(2)).squeeze(-1)  # (b, t) after squeeze
        kl_div = 0.05 * (base_log_probs - log_probs)  # (b, t)
        # add the reward to the last token
        kl_div[:, -1] += reward

        return kl_div, reward

    def step_episode(self, state: str) -> Tuple[
        list[str], torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Do a full rollout. That is generate a sequence of length max_length or until the
        end of the sequence is reached. Return a list of tuples of the form (state, reward).
        """

        # tokenize the state
        idx = self.tokenizer(state, return_tensors="pt").input_ids.to(self.device)  # (b, t)
        # generate a sequence of length max_length or until the end of the sequence is reached
        self.model.eval()
        generated_idx, logits, values = self.model.generate(self.tokenizer, idx, max_new_tokens=self.max_length,
                                                            temperature=1.0, do_sample=True, top_k=None,
                                                            stop_words=self.stop_words)
        self.model.train()
        # cut the last generated_idx
        generated_idx = generated_idx[:, :-1]  # (b, t)
        # make the mask. 1 if the token is generated, 0 if it was already part of the prompt
        batch_size, total_length = generated_idx.shape

        completion_sequence_length = total_length - idx.shape[1]
        # Create the prompt mask (zeros)
        prompt_mask = torch.zeros(idx.shape)  # (b, t)

        # Create the completion mask (ones)
        completion_mask = torch.ones((batch_size, completion_sequence_length))  # (b, t')

        # Concatenate the masks along the second dimension (axis 1)
        mask = torch.cat((prompt_mask, completion_mask), dim=1).to(self.device)  # (b, t)
        # split the sequence into individual states
        rewards, raw_reward = self.calculate_reward(generated_idx, logits)  # (b, t), (b,)
        # print token and the reward (.2 digits)
        # print([(self.tokenizer.decode(generated_idx[0, i]), "{:.2f}".format(rewards[0, i].item())) for i in range(rewards.shape[1])])
        # create a list of tuples of the form (state, reward, done)
        decoded_gen = self.tokenizer.batch_decode(generated_idx)  # (b,)
        return decoded_gen, generated_idx, rewards, values, logits, raw_reward, mask


sentiment_fn = pipeline(
    "sentiment-analysis",
    "lvwerra/distilbert-imdb",
    top_k=2,
    truncation=True,
    batch_size=256,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reward_func(x):
    sentiments = sentiment_fn(x)
    results = []
    for labeled_data in range(len(sentiments)):
        for label in sentiments[labeled_data]:
            if label["label"] == "POSITIVE":
                results.append(torch.tensor(label["score"]))

    results = torch.stack(results)
    results = results
    return results


# model = GPT.from_pretrained("lvwerra/gpt2-imdb").to(device)
model = GPT.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# add pad token
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

optimizer = optim.Adam(model.parameters(), lr=0.0003)

positive_reviews = [
    "Captivating and deeply moving, The Shawshank Redemption is a masterpiece of storytelling that explores the depths of the human spirit. Director Frank Darabont brings Stephen King's novella to life, painting a vivid picture of the brutality and hope within the walls of Shawshank Penitentiary. Tim Robbins and Morgan Freeman deliver unforgettable performances as Andy Dufresne and Red, two prisoners who forge an unbreakable bond despite their circumstances. Ultimately, it's a story of redemption and the power of hope that resonates with audiences long after the credits roll.",
    "Francis Ford Coppola's epic crime drama is a cinematic tour de force that remains an undisputed classic. With a powerful ensemble cast led by Marlon Brando, Al Pacino, and James Caan, The Godfather delves into the inner workings of the Mafia and the Corleone family. Filled with intense drama, gripping suspense, and unforgettable characters, this film is a testament to the power of storytelling and the timeless themes of loyalty, power, and family.",
    "Quentin Tarantino's groundbreaking crime drama revolutionized the genre with its non-linear narrative, razor-sharp dialogue, and unforgettable characters. John Travolta, Samuel L. Jackson, and Uma Thurman deliver iconic performances as they navigate the seedy underbelly of Los Angeles. Pulp Fiction is a wild, unpredictable ride that continues to captivate audiences with its blend of dark humor, violence, and pop culture references.",
    "Steven Spielberg's heart-wrenching depiction of the Holocaust is a masterpiece of historical storytelling. Shot in black and white, the film follows the efforts of Oskar Schindler (Liam Neeson) as he saves over a thousand Jews from the clutches of the Nazis. Schindler's List is a powerful testament to the resilience of the human spirit and the importance of remembering the atrocities of the past.",
    "Christopher Nolan's second installment in his Batman trilogy is a gripping, action-packed masterpiece that redefined the superhero genre. With Heath Ledger's chilling portrayal of the Joker, The Dark Knight is a thrilling exploration of chaos and morality. Christian Bale's nuanced performance as Batman raises the stakes, and the film's stunning visuals and unforgettable score make it a must-watch for any cinephile.",
    "Tom Hanks delivers one of his most memorable performances as the lovable, simple-minded Forrest Gump. This heartwarming tale of love, friendship, and perseverance takes audiences on a journey through American history, as seen through the eyes of an unlikely hero. With its powerful storytelling and unforgettable characters, Forrest Gump is a timeless classic that will leave you laughing, crying, and feeling inspired.",
    "This visually stunning and emotionally resonant film by director Michel Gondry is a unique exploration of love, memory, and the pain of heartbreak. Jim Carrey and Kate Winslet deliver unforgettable performances as two strangers who meet by chance and fall in love. Eternal Sunshine of the Spotless Mind is a poignant, thought-provoking film that will leave you feeling inspired and hopeful.",
    "Jonathan Demme's psychological thriller is a chilling and masterfully crafted exploration of the darker corners of the human psyche. Jodie Foster's powerful performance as FBI trainee Clarice Starling, paired with Anthony Hopkins' terrifying portrayal of cannibalistic serial killer Dr. Hannibal Lecter, creates an electrifying dynamic that keeps audiences on the edge of their seats. The Silence of the Lambs is a haunting, suspenseful masterpiece that remains one of the most iconic films in the thriller genre.",
    "David Fincher's adaptation of Chuck Palahniuk's novel is a daring, thought-provoking exploration of consumerism, masculinity, and the search for identity in modern society. Brad Pitt and Edward Norton deliver outstanding performances as Tyler Durden and the unnamed narrator, whose lives become entwined in a dangerous and anarchic underground movement. Fight Club is a visually striking, darkly humorous, and deeply unsettling film that continues to resonate with audiences two decades after its release.",
    "Christopher Nolan's mind-bending sci-fi thriller takes audiences on a thrilling journey through the world of dreams and the subconscious mind. Leonardo DiCaprio leads an all-star cast in a complex, visually stunning narrative that challenges the boundaries of reality. The film's intricate plot, mesmerizing visual effects, and unforgettable score make Inception an unforgettable cinematic experience that demands multiple viewings to fully appreciate its brilliance."
]
mixed_reviews = [
    "The Matrix Reloaded (2003): While the sequel to the groundbreaking The Matrix offers impressive action sequences and stunning visual effects, it falls short in terms of storytelling and character development. The convoluted plot and philosophical discussions can be confusing, but fans of the original may still find enjoyment in the expansion of the Matrix universe.",
    "Avatar (2009): Visually stunning and groundbreaking in terms of its use of 3D technology, James Cameron's Avatar offers a feast for the eyes. However, the film suffers from a derivative plot and underdeveloped characters, leaving some viewers more impressed with its technical achievements than its storytelling.",
    "The Hobbit: An Unexpected Journey (2012): Peter Jackson's return to Middle-earth is a visually impressive spectacle, but its bloated running time and uneven pacing make it a less satisfying experience than the beloved Lord of the Rings trilogy. While fans of the source material may appreciate the attention to detail, others may find it difficult to stay engaged throughout the film's nearly three-hour runtime.",
    "Jurassic World (2015): This long-awaited return to the Jurassic Park franchise offers thrilling dinosaur action and impressive visual effects but falls short in terms of character development and originality. While the film is undeniably entertaining, it relies heavily on nostalgia and fails to recapture the magic of the original.",
    "Ocean's Twelve (2004): While the star-studded ensemble cast and stylish direction make for an entertaining caper, Ocean's Twelve suffers from a convoluted plot and a lack of the charming banter that made the first film so enjoyable. Fans of the original may still find enjoyment in the slick heist sequences and European locations, but the film ultimately feels like a missed opportunity.",
    "Spectre (2015): The fourth installment in the Daniel Craig era of James Bond films offers thrilling action and stunning visuals but is weighed down by a convoluted plot and an underwhelming villain. Despite the film's shortcomings, Craig's performance as Bond remains a highlight, and die-hard fans may still appreciate the nods to the franchise's history.",
    "The Hangover Part II (2011): Essentially a rehash of the original, The Hangover Part II delivers more of the raunchy humor and outrageous situations that made the first film a hit. While some audience members may enjoy the familiarity, others may be disappointed by the lack of originality and the film's reliance on shock value to generate laughs.",
    "Transformers: Age of Extinction (2014): Michael Bay's fourth installment in the Transformers franchise offers more of the trademark explosive action and impressive visual effects but fails to inject any new life into the series. The thin plot, juvenile humor, and excessive running time make it a difficult watch for anyone but the most devoted fans of the franchise.",
    "Suicide Squad (2016): An ensemble of intriguing characters and a promising premise are let down by a weak plot, choppy editing, and an underwhelming villain. While Margot Robbie's performance as Harley Quinn and the film's stylish visuals are highlights, Suicide Squad ultimately fails to live up to its potential as a fresh take on the superhero genre.",
    "Indiana Jones and the Kingdom of the Crystal Skull (2008): Nostalgia and the return of Harrison Ford as the iconic archaeologist are not enough to save this fourth installment in the Indiana Jones franchise. The film's outlandish plot, excessive CGI, and uneven pacing make it a far cry from the charm and excitement of the original trilogy."]

negative_reviews = [
    "Battlefield Earth (2000): This notorious sci-fi disaster is marred by a convoluted plot, laughable dialogue, and a bizarre performance by John Travolta. The film's overuse of Dutch angles and poor special effects only add to the overall sense of confusion and disorientation, making it a difficult watch for even the most devoted fans of the genre.",
    "The Emoji Movie (2017): This animated feature suffers from a lack of originality, wit, and charm, relying on tired tropes and clichés to tell its story. The film's attempt to capitalize on the popularity of emojis feels like a blatant cash grab, and the end result is a forgettable and uninspired viewing experience.",
    "Batman & Robin (1997): This infamous installment in the Batman franchise is a far cry from the darker, more serious tone of its predecessors. With its campy humor, over-the-top performances, and outlandish costumes, Batman & Robin feels more like a parody than a genuine superhero film. The film's lack of cohesion and focus make it a disappointing entry in the franchise.",
    "Catwoman (2004): Halle Berry's turn as the feline anti-hero is marred by a weak script, poor direction, and underwhelming special effects. The film's lack of a cohesive plot and character development make it a wasted opportunity to explore the potential of this iconic character.",
    "Gigli (2003): This notorious romantic comedy, starring Ben Affleck and Jennifer Lopez, is a prime example of on-screen chemistry gone wrong. The film's weak script, poor direction, and lack of humor make it a painful viewing experience that is best left forgotten.",
    "The Last Airbender (2010): M. Night Shyamalan's adaptation of the beloved animated series is a crushing disappointment for fans and newcomers alike. With its poor pacing, wooden performances, and lackluster special effects, The Last Airbender fails to capture the magic and depth of the original show.",
    "Jack and Jill (2011): Adam Sandler's attempt at playing both lead roles in this comedy falls flat, with neither character proving to be particularly funny or engaging. The film's reliance on crude humor and tired gags make it a forgettable and unsatisfying experience.",
    "Movie 43 (2013): This anthology of comedy shorts boasts an impressive ensemble cast, but the end result is a disjointed and offensive mess. The film's reliance on shock value and crude humor over genuine wit and cleverness make it a poor showcase for the talents of its actors.",
    "Dragonball Evolution (2009): This live-action adaptation of the beloved anime and manga series is a major misfire, with poor casting choices, unimpressive special effects, and a complete disregard for the source material. Fans of the original series will be left disappointed and frustrated by this lackluster effort.",
    "Fifty Shades of Grey (2015): This adaptation of the best-selling novel fails to deliver on the steamy, titillating promise of its source material. With wooden performances, stilted dialogue, and a lack of chemistry between the leads, Fifty Shades of Grey is a bland and uninspired attempt at bringing the controversial story to the big screen."]

training_examples = positive_reviews + negative_reviews + mixed_reviews

# we need to finetune only the "model" (not the base model) on some examples first before we begin RL training
# train model on the training examples
for i in range(10):
    optimizer.zero_grad()
    random.shuffle(training_examples)
    for example in training_examples:
        # print(example)
        idx = tokenizer(example, return_tensors="pt").input_ids.to(device)
        target = idx[:, 1:].contiguous()
        idx = idx[:, :-1].contiguous()
        logits, _, _ = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1), ignore_index=tokenizer.pad_token_id)

        loss.backward()
    optimizer.step()
    print("SFT loss", loss.item())

prompts = ["I don't know much about Good Will Hunting but"]

max_length = 80

# base model is the model but copied and frozen
base_model = copy.deepcopy(model)
base_model.eval()
base_model.to(device)

env = LMEnv(model, base_model, tokenizer, max_length, reward_func, prompts, [], device)


def evaluate_policy():
    total_reward = []
    imdb = load_dataset("imdb", split="train+test")

    eval_prompts = [" ".join(review.split()[:4]) for review in imdb["text"]][:100]

    for i in range(len(eval_prompts)):
        # generate with model
        generated_sequence_idx, _, _ = model.generate(tokenizer,
                                                      tokenizer(eval_prompts[i], return_tensors="pt").input_ids.to(
                                                          device), max_new_tokens=max_length)
        # get reward
        reward = reward_func(tokenizer.batch_decode(generated_sequence_idx))
        total_reward.append(reward)

    return torch.concat(total_reward)  # (100, 1)


optimizer = optim.Adam(model.parameters(), lr=3e-4)

total_timesteps = 0
episodes = 20000
lam = 0.95
gamma = 1
ppo_epochs = 1
ppo_clip = 0.2
virtual_batch = 128
# total_rewards = []
total_raw_rewards = []
for episode in range(episodes):
    # print("Episode", episode, "of", episodes)
    state = env.reset()
    if len(state) > 900:
        # print("Skipping episode because it's too long")
        continue
    episode_data = env.step_episode(state)
    decoded_str, actions, rewards, state_values, logits, raw_reward, mask = episode_data
    # decoded_str is of (B) length
    # actions is of (B, T) shape
    # rewards is of (B, T) shape
    # state_values is of (B, T, 1) shape
    # logits is of (B, T, V) shape
    # raw_reward is of (B,) shape
    # mask is of (B, T) shape

    completion_pos = torch.argmax(mask, dim=1)  # B, 1
    # cut actions only from completion_pos onwards
    actions = actions[:, completion_pos[0]:]  # TODO: fix for proper batch support
    rewards = rewards[:, completion_pos[0]:]
    state_values = state_values[:, completion_pos[0]:]
    logits = logits[:, completion_pos[0]:]

    old_log_probs = F.log_softmax(logits, dim=-1)
    # get the log probs only for the actions that were taken
    old_log_probs = torch.gather(old_log_probs, 2, actions.unsqueeze(1)).squeeze(1)
    # total_rewards.append(rewards.detach().cpu().numpy())
    total_raw_rewards.append(raw_reward.detach().cpu().numpy())

    advantages = []
    gae = 0
    for i in reversed(range(actions.shape[1])):
        with torch.no_grad():
            next_value = state_values[:, i + 1] if i != actions.shape[1] - 1 else 0.0  # B, 1
        delta = (rewards[:, i] + gamma * next_value - state_values[:, i])  # B, 1
        gae = delta + gamma * lam * gae  # B, 1
        advantages.append(gae)
    advantages = advantages[::-1]
    advantages = torch.stack(advantages, dim=1)  # B, T, 1
    # normalize the advantages
    returns = advantages + state_values  # (B, T, 1) + (B, T, 1) = (B, T, 1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.squeeze(-1).detach()  # B, T
    returns = returns.squeeze(-1).detach()  # B, T

    for _ in range(ppo_epochs):
        total_timesteps += 1
        # for the states in the past, run the policy to get the new action dist
        model.eval()
        new_logits_actions, _, new_values = model(actions)
        model.train()
        new_log_probs = F.log_softmax(new_logits_actions, dim=-1)
        # get log probs only for the tokens that were generated
        new_log_probs = torch.gather(new_log_probs, 2, actions.unsqueeze(1)).squeeze(1)  # B, T
        # print the value for each token (token, value)
        # print([f"{tokenizer.decode([token])}={'{:.2f}'.format(value.item())}" for token, value in zip(actions[0], new_values[0])])
        # print the advantages the same way
        # print([f"{tokenizer.decode([token])}={'{:.2f}'.format(adv.item())}" for token, adv in zip(actions[0], advantages[0])])

        ratio = (new_log_probs - old_log_probs.detach()).exp()  # B, T

        obj1 = ratio * advantages  # (B, T) * (B, T) = (B, T)
        obj2 = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * advantages
        policy_loss = -torch.min(obj1, obj2).mean()

        value_loss = F.mse_loss(new_values.squeeze(-1), returns)  # (B, T) - (B, T) = (B, T)

        loss = policy_loss + 0.5 * value_loss
        loss /= virtual_batch

        # mimick batching using gradient accumulation
        if total_timesteps % virtual_batch == 0:
            optimizer.zero_grad()
        loss.backward()
        if total_timesteps % virtual_batch == virtual_batch - 1:
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    if episode % virtual_batch == 0:
        print("Episode: ", episode, "\nMean final reward", np.mean(np.array(total_raw_rewards)), "\nPolicy Loss: ",
              policy_loss.item(), "\nValue Loss: ", value_loss.item(), "\nLoss: ", loss.item() * 128, "\nMean Returns",
              returns.mean().item(), "\nMean Advantages", advantages.mean().item(), "\nEvaluated mean reward",
              evaluate_policy().mean().item(), "\n\n")
        print(decoded_str)
        # "\nMean Total Reward: ", np.mean(np.concatenate(total_rewards, axis=1)),
