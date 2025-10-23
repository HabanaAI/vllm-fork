#curl http://localhost:8295/v1/completions   -H "Content-Type: application/json"   -d '{
#      "model": "/software/data/pytorch/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/",
#            "prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.",
#	           "max_tokens": 100,
#		          "temperature": 0
#			      }'

 curl http://localhost:1295/v1/completions   -H "Content-Type: application/json"   -d '{
       "model": "/mnt/weka/data/llm-d-models-pv/Meta-Llama-3.1-8B-Instruct/",
             "prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.",
 	           "max_tokens": 100,
 		          "temperature": 0
 			      }'



# curl http://localhost:9909/v1/completions   -H "Content-Type: application/json"   -d '{
#       "model": "/software/data/pytorch/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/",
#             "prompt": "Mark Elliot Zuckerberg is an American businessman who co-founded the social media service Facebook and its parent company Meta Platforms, of which he is the chairman, chief executive officer, and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy. Born in White Plains, New York, Zuckerberg briefly attended Harvard College, where he launched Facebook in February 2004 with his roommates Eduardo Saverin, Andrew McCollum, Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest self-made billionaire[a] in 2008, at age 23, and has consistently ranked among the worlds wealthiest individuals. According to Forbes, Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025, making him the second-richest individual in the world.",
# 	           "max_tokens": 100,
# 		          "temperature": 0
# 			      }'
