import os

import dotenv
dotenv.load_dotenv()
# General bot settings to use Pro settings you need to download Pro version from: www.automated-bots.com

#PRO FEATURE - browser you want the bot to run ex: ["Chrome"] or ["Firefox"]. Firefox is only supported in Pro feature
browser = ["Chrome"]
# Enter your Linkedin password and username below. Do not commit this file after entering these credentials.
# Linkedin credentials
email = os.getenv("email")
password = os.getenv("password")

#PRO FEATURE - Optional! run browser in headless mode, no browser screen will be shown it will work in background.
headless = False
#PRO FEATURE - Optional! If you left above credentials fields empty. For Firefox or Chrome enter profile dir to run the bot to prevent logging in your account each time
# get Firefox profile path by typing following url: about:profiles
firefoxProfileRootDir = r""
# get Chrome profile path by typing following url: chrome://version/
chromeProfilePath = r""

# These settings are for running Linkedin job apply bot.
# location you want to search the jobs - ex : ["Poland", "Singapore", "New York City Metropolitan Area", "Monroe County"]
# continent locations:["Europe", "Asia", "Australia", "NorthAmerica", "SouthAmerica", "Africa", "Australia"]
location = [
    "London",
    # "Berlin",
    # "Amsterdam",
    # "Dublin",
    # "Europe"
]
# keywords related with your job search
keywords = [
    "iOS",
    "Swift",
    "python",
    # "backend",
    # "devops",
    # "programming",
    # "software"
]

# https://www.linkedin.com/jobs/search/?currentJobId=4124758757&f_AL=true&f_E=3%2C4&f_SB2=44&f_TPR=r2592000&f_WT=3&origin=JOB_SEARCH_PAGE_JOB_FILTER
#job experience Level - ex:  ["Internship", "Entry level" , "Associate" , "Mid-Senior level" , "Director" , "Executive"]
experienceLevels = [ "Mid-Senior level", "Senior level"]
#job posted date - ex: ["Any Time", "Past Month" , "Past Week" , "Past 24 hours"] - select only one
# datePosted = ["Past Week"]
# datePosted = ["Past 24 hours"]
datePosted = ["Past 24 hours"]
#job type - ex:  ["Full-time", "Part-time" , "Contract" , "Temporary", "Volunteer", "Intership", "Other"]
jobType = ["Full-time"]
#remote  - ex: ["On-site" , "Remote" , "Hybrid"]
remote = ["On-site", "Hybrid", "Remote"]
#salary - ex:["$40,000+", "$60,000+", "$80,000+", "$100,000+", "$120,000+", "$140,000+", "$160,000+", "$180,000+", "$200,000+" ] - select only one
salary = [ "$50,000+"]
#sort - ex:["Recent"] or ["Relevent"] - select only one
sort = ["Relevent"]
#Blacklist companies you dont want to apply - ex: ["Apple","Google"]
blacklistCompanies = ["Kraken", "Harrington Starr","Apleona Combined Technical Solutions", "Soundmouse", "Orfium", "Octopus"]
#Blaclist keywords in title - ex:["manager", ".Net"]
blackListTitles = ["Ruby", "Lawyer", "Apprenticeship", "Designer"]
#Follow companies after sucessfull application True - yes, False - no
followCompanies = False
#Below settings are for linkedin bot Pro, you can purchase monthly or yearly subscription to use them from me.
#If you have multiple CV's you can choose which one you want the bot to use. (1- the first one on the list, 2 - second , etc)
preferredCv = 1
#PRO FEAUTRE! - Output unaswered questions into a seperate text file, will output radio box, dropdown and input field questions into seperate .yaml file
outputSkippedQuestions = True
#Use AI to fill and answer skipped questions. Will cost 5 credits per answer cause of computational power.
useAiAutocomplete = True
#Only Apply these companies -  ex: ["Apple","Google"] -  leave empty for all companies 
onlyApplyCompanies = []
#Only Apply titles having these keywords -  ex:["web", "remote"] - leave empty for all companies 
onlyApplyTitles = [] 
#PRO FEATURE! - Dont apply the job posted by the Hiring member contains this in his/her name - ex: ["adam","Sarah"]
blockHiringMember = [] 
#PRO FEATURE! - Only apply the job sposted by the Hiring member contains this in his/her name - ex: ["adam","Sarah"]
onlyApplyHiringMember = [] 
#PRO FEATURE! - Only apply jobs having less than applications - ex:["100"] will apply jobs having upto 100 applications 
onlyApplyMaxApplications = []
#PRO FEATURE! - Only apply jobs having more than applications - ex:["10"] will apply jobs having more than 10 applications 
onlyApplyMinApplications = []
#PRO FEATURE! - Only apply jobs having these keywords in the job description
onlyApplyJobDescription = []
#PRO FEATURE! - Do not apply the jobs having these keywords in the job description
blockJobDescription = []
#PRO FEATURE! - Apply companies having equal or more than employes - ex: ["100"]
onlyAppyMimEmployee = []
#PRO FEATURE - Apply the ones linkedin is saying "you may be a goodfit"
onlyApplyLinkedinRecommending = False
#PRO FEATURE - Only apply the ones you have skilled badge
onlyApplySkilledBages = False
#PRO FEATURE! - Save the jobs by pressing SAVE button before apply  True - yes, False - no
saveBeforeApply = False
#PRO FEATURE! - Sent a message to the hiring manager once you apply for the role
messageToHiringManager = ""
#PRO FEATURE! - List and output non Easy Apply jobs links
listNonEasyApplyJobsUrl = False
#PRO FEATURE! - Select radio button for unsawered questions. If the bot cannot find an answer for a radio button, it will automatically select first or second option. Default radio button answer, 1 for Yes, 2 for No. Leave empty if you dont want this option.
defaultRadioOption = 1
#PRO FEATURE! - Check yes or no to all checkbox questions (True - yes, False - no), leave empty if you dont want this option
answerAllCheckboxes = ""
#PRO FEAUTRE! - Output file type. Can be .txt or .csv (excel) 
outputFileType = [".txt"]

# These settings are for running AngelCO job apply bot you need to purchase AngelCo bot obtain bot password, paste below and then run the bot.
AngelCoBotPassword = ""
# AngelCO credits
AngelCoEmail = ""
AngelCoPassword = ""
# jobTitle ex: ["Frontend Engineer", "Marketing"]
angelCoJobTitle = ["Frontend Engineer"]
# location ex: ["Poland"]
angelCoLocation = ["Poland"]

# These settings are for running GlobalLogic job apply bot you need to purchase GlobalLogic bot obtain bot password, paste below and then run the bot.
GlobalLogicBotPassword = ""
# AngelCO credits
GlobalLogicEmail = ""
GlobalLogicPassword = ""
# Functions ex: ["Administration", "Business Development", "Business Solutions", "Content Engineering", 	
# Delivery Enablement", Engineering, Finance, IT Infrastructure, Legal, Marketing, People Development,
# Process Management, Product Support, Quality Assurance,Sales, Sales Enablement,Technology, Usability and Design]
GlobalLogicFunctions = ["Engineering"]
# Global logic experience: ["0-1 years", "1-3 years", "3-5 years", "5-10 years", "10-15 years","15+ years"]
GlobalLogicExperience = ["0-1 years", "1-3 years"]
# Global logic location filter: ["Argentina", "Chile", "Crotia", "Germany", "India","Japan", "Poland"
# Romania, Sweden, Switzerland,Ukraine, United States]
GlobalLogicLocation = ["poland"]
# Freelance yes or no
GlobalLogicFreelance = ["no"]
# Remote work yes or no
GlobalLogicRemoteWork = ["yes"]
# Optional! Keyword:["javascript", "react", "angular", ""]
GlobalLogicKeyword = ["react"]
# Global Logic Job apply settinngs
FirstName = "O"
LastName = "D"
Email = "asdsa@gmail.com"
LinkedInProfileURL = "www.google.com"
Phone = "" #OPTIONAL
Location = "" #OPTIONAL
HowDidYouHeard = "" #OPTIONAL
ConsiderMeForFutureOffers = True #true = yes, false = no

 # Testing & Debugging features
displayWarnings = True