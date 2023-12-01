---
toc: false
layout: post
description: Now, can I let AI do all the work?
categories: [markdown, chat]
title: ChatGPT All the Way Down
---

In two previous posts (see [here](https://bobowedge.github.io/adventures-in-telework/pytorch/jupyter/2020/11/12/chat_generation.html) 
and [here](https://bobowedge.github.io/adventures-in-telework/jupyter/2021/02/02/chat_bugaloo.html)), I used PyTorch and some
tools from the [fastai course](https://course.fast.ai/) that I was taking to try to generate some fake chat. The corpus
that I used was from a private Google Chat (nee Hangouts) chatroom that I was in. I was trying to generate some chat that
at least somewhat resemble the messages that we sent in the room. With mixed result, I'd say.

Of course, this was all done before most people (including myself) were aware of ChatGPT. So, now, naturally, the thing to do is 
let ChatGPT loose on the input chat and see what it generates. As before, there's a warning:

> Warning: Some of the chat content may contain profanity or stupidity.

For the record, I used ChatGPT version 3.5.

For me, it took a couple of iterations of input into ChatGPT to get it to understand what exactly I wanted and to actually
generate some chat.  In particular, I learned that there's a limit on how many input lines from the original chat it could
process.  I also made sure to give it an idea of the structure of the input data. Here's the prompt that I ended up using:

```Generate the next 50 lines of this chat, where the sender is listed first on each line, followed by ::, followed by the message. Also, images sent have been replaced by the token <MEME>.```

Then I provided it with the first 500 lines extracted from the Hangouts data that I had.  Here's what ChatGPT came up with:

```
Kappa :: Just RSVPing in, what's up?
Omega :: Did it go down? Did you see the latest AliExpress find?
Kappa :: I couldn't see it and my messages weren't sending.
Omega :: <MEME>
Kappa :: I've gotta have that!
Gamma :: Looks like I'm flooding this chat at work.
Omega :: Gotta keep it safe for work?
Psi :: Nope.
Omega :: I'm hunting down all those AliExpress rubber finds.
Kappa :: Technically, they blur the nipples, so it's not nudity.
Kappa :: Curse those Allo puritans!
Kappa :: Allo's got me trundling to my phone for messages.
Kappa :: Productivity's hit rock bottom.
Gamma :: More frequent, less quality shit posting, brace yourselves.
Psi :: GG
Kappa :: <MEME>
Kappa :: What if Google paid us for each shitpost?
Psi :: Soon, Hangouts will fall too.
Kappa :: Cha-ching!
Beta :: The Gamma/Omega drama continues! Gamma snubbed invites from Omega and me three times.
Beta :: <MEME>
Beta :: Hangouts is the worst.
Kappa :: Weirdly itchy night, resorted to baby powder.
Kappa :: <MEME>
Omega :: Hangouts is trash.
Kappa :: Standard messaging now supports GIFs.
Psi :: Why's Hangouts so awful?
Psi :: Humble <MEME> perhaps?
Omega :: I bet Psi says GG to his dumps before flushing.
Gamma :: Who doesn't, honestly?
Beta :: Kappa's got some southern phrase that sounds racist, even if it isn't.
Beta :: Gamma, you literally crap on your own crap.
Beta :: Omega: "Don't worry, crap, you're still better than Anthem."
Omega :: Ding!
Gamma :: Brutal but fair.
Gamma :: Breshad Perriman wants out of the Browns now that they've got OBJ.
Gamma :: Good luck, loser.
Psi :: GG
Kappa :: <MEME>
Kappa :: Allo's missing the 'go to last read' option. Scroll, scroll, scroll.
Gamma :: Wow, old-fashioned racism there.
Kappa :: <MEME>
Psi :: <MEME>
Kappa :: The dead horse is ablaze.
Kappa :: Bushmaster ad: <MEME>
Kappa :: That lawsuit refers to the one on the left, circa 2010.
Psi :: Boo, secret chats I'm not part of anymore.
Kappa :: Gammanade!
Kappa :: The CT supreme court greenlights the lawsuit against Remington and the retailer who sold Lanza's Mom the AR 15. It's based on their marketing violating CT state law, not covered by the 2005 federal law shielding gun manufacturers from liability.
Kappa :: You're up to date now.
Kappa :: Or join Synergist and get back in our chat.
Gamma :: That was a solid grenade, got Kappa going.
Kappa :: I monologued for ages.
Kappa :: <MEME>
Omega :: Mentally challenged
Omega :: <MEME>
Kappa :: There's no more Ted Cruz, only Reek.
Omega :: V was explaining mutations, I had to stop her unless she could do a Patrick Stewart voiceover. Couldn't follow otherwise.
Gamma :: Lol
Gamma :: <MEME>
Psi :: Deer roam my neighborhood often. How urban does it need to be?
Kappa :: Social commentary on Columbia!
Omega :: Feels post-apocalyptic already.
Omega :: Embrace the post-apocalyptic!
Gamma :: But none of you have died to be eaten yet!
Kappa :: Trying!
Kappa :: <MEME>
Omega :: <MEME>
Beta :: Needs a consent line.
Kappa :: Lol
Kappa :: The art reminds me of the Ctrl Alt Del webcomic. Random, right?
Kappa :: <MEME>
Psi :: I'd complain, but I feel Kappa might explode <MEME>
Kappa :: Hmm...
Kappa :: Seems like a joke bill.
Kappa :: "To start a conversation."
Psi :: Alright, I'll bite. What's the equally dangerous entity he's arguing ad absurdum about?
Kappa :: ?
Kappa :: He introduced it and will walk it back.
Kappa :: Then claim on Fox how great he is.
Gamma :: Can't even have FFS as a clan tag in Div2. Literally unplayable.
Psi :: JFC
Kappa :: Also, a good tag name.
Omega :: Jewish Fried Chicken!
Kappa :: <MEME>
Kappa :: Spotted a truck with the Destiny logo in the back. Neat!
Omega :: Div2, Gamma?
Gamma :: I'll be back home in half an hour, just in case.
Gamma :: Trump's tweet about Paris Accord and Yellow Vest Protesters is... puzzling.
Omega :: Mentally challenged
Psi :: What list is he talking about?
Gamma :: All the lists!
Kappa :: The "Top 10 Coolest Country" list, like Forbes.
Psi :: export COOL_COUNTRY="USA, USA, USA, USA, USA" ; while true ; do echo ${COOL_COUNTRY} done
Kappa :: Lol
Gamma :: Dammit, my MoCa adapter is busted!
Kappa :: 😞
Kappa :: Ahead of us in culture: <MEME>
Omega :: 80085 must have been taken.
Omega :: Damage scales, armor doesn't. We're close in level, so not too bad at our current state.
Omega :: Probably why we sucked so much that first night.
Omega :: Blame Gamma and his free time!
Sigma :: This game is tougher than I remember.
Kappa :: <MEME>
Psi :: <MEME>
Omega :: Wonder if there's a TDY to NZ?
Psi :: Maybe not for you.
Kappa :: Lol
Kappa :: <MEME>
Omega :: Second!
Gamma :: <MEME>
Kappa :: Relatable.
Kappa :: Greta Wolf Lodge is a furry breeding ground disguised as an indoor water park for kids.
```

Honestly, I'm both impressed and a bit disappointed by this output. I think it does mostly capture the chat, but it accomplishes
that by just regurgitating exact lines from the chat.  For example, last 10 lines that were output are an exact match to 
some of the lines of input (complete with the typo 'Greta').  To be fair, I'm sure we repeat a lot of the same lines (and 
dub it a 'callback'), but I think we have more variety in our chats.

Of course, I told ChatGPT all of this, in so many words, so it gladly returned a batch with more lines of chat with "some 
lively conversations":

```
Kappa :: Hey, did you catch the latest episode of that show we were discussing?
Omega :: Not yet, but I heard it was wild. Is it worth binging?
Kappa :: Absolutely! It's like they crammed a year's worth of plot into one episode.
Gamma :: Sounds like my kind of chaos.
Psi :: Speaking of chaos, I accidentally sent my boss a cat meme yesterday.
Omega :: Classic! How did they respond?
Psi :: Surprisingly, they replied with a dog meme. Boss level: Expert.
Beta :: I wish my boss had that kind of sense of humor.
Kappa :: Have you seen the new tech gadget everyone's raving about?
Gamma :: Yeah, it's like they took something simple and made it unnecessarily complex.
Omega :: That's the tech industry in a nutshell.
Psi :: Remember when phones were just for calling people?
Kappa :: Now they run the world and occasionally make phone calls.
Gamma :: Speaking of tech, did you see that AI-generated art collection?
Beta :: It's fascinating how AI is delving into creativity.
Psi :: Imagine an AI joining our chat. That'd be an adventure.
Kappa :: We'd have to teach it our meme game.
Omega :: AI mastering memes? I'm both intrigued and slightly terrified.
Gamma :: We'd create the ultimate meme machine!
Psi :: I think the internet might explode from that much humor.
Kappa :: But what a way to go! Boom, meme overload!
Omega :: A meme-fueled apocalypse would be the most unexpected twist.
Beta :: Memes as the final frontier of human achievement—what a legacy!
Gamma :: The future textbooks would have an entire chapter on meme culture.
Psi :: The art of laughter—our contribution to history.
Kappa :: Maybe one day, we'll all be immortalized in memes.
Omega :: I can see it now: "Legends of the Internet: The Meme Masters."
Gamma :: Our digital avatars gracing the halls of internet fame.
Beta :: And our wit echoing through the corridors of cyberspace!
Psi :: We'd go down in history as the pioneers of memeology.
Kappa :: A toast to eternal internet stardom!
Omega :: May our memes be forever dank and our humor everlasting!
Gamma :: Long live the meme dynasty!
Psi :: Cheers to the legends we're destined to become!
Beta :: Here's to laughter, friendship, and the power of memes!
```

This might be a conversation that some people might have (maybe in a sitcom?), but I can confidently say that this does
not fit the timbre of our chatroom. :stuck_out_tongue:

For anyone who made it this far, I couldn't find any good relevant memes about Great (or Greta) Wolf Lodge, but here are 
two of the images I got when I searched [Simpsons screencaps](https://frinkiac.com/) for 'Great Wolf Lodge'. 

![Homer on Ghost Pepper]({{site.baseurl}}/images/simpsons_wolf.jpg){:class="img-responsive"}

![Ranier Wolfcastle]({{site.baseurl}}/images/simpsons_wolf2.jpg){:class="img-responsive"}