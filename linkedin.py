import time,math,random,os

import selenium.common.exceptions

import utils,constants,config
import pickle, hashlib

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

import yaml

def load_additional_questions(filepath="AdditionalQuestions.yaml"):
    with open(filepath, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data

class Linkedin:
    def __init__(self):
            utils.prYellow("🤖 Thanks for using Easy Apply Jobs bot, for more information you can visit our site - www.automated-bots.com")
            utils.prYellow("🌐 Bot will run in Chrome browser and log in Linkedin for you.")
            self.driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),options=utils.chromeBrowserOptions())
            self.cookies_path = f"{os.path.join(os.getcwd(),'cookies')}/{self.getHash(config.email)}.pkl"
            self.driver.get('https://www.linkedin.com')
            self.loadCookies()
            self.answers = load_additional_questions()

            if not self.isLoggedIn():
                self.driver.get("https://www.linkedin.com/login?trk=guest_homepage-basic_nav-header-signin")
                utils.prYellow("🔄 Trying to log in Linkedin...")
                try:    
                    self.driver.find_element("id","username").send_keys(config.email)
                    time.sleep(2)
                    self.driver.find_element("id","password").send_keys(config.password)
                    time.sleep(2)
                    self.driver.find_element("xpath",'//button[@type="submit"]').click()
                    time.sleep(30)
                except:
                    utils.prRed("❌ Couldn't log in Linkedin by using Chrome. Please check your Linkedin credentials on config files line 7 and 8.")

                self.saveCookies()
            # start application
            self.linkJobApply()

    def getHash(self, string):
        return hashlib.md5(string.encode('utf-8')).hexdigest()

    def loadCookies(self):
        if os.path.exists(self.cookies_path):
            cookies =  pickle.load(open(self.cookies_path, "rb"))
            self.driver.delete_all_cookies()
            for cookie in cookies:
                self.driver.add_cookie(cookie)

    def saveCookies(self):
        pickle.dump(self.driver.get_cookies() , open(self.cookies_path,"wb"))
    
    def isLoggedIn(self):
        self.driver.get('https://www.linkedin.com/feed')
        try:
            self.driver.find_element(By.XPATH,'//*[@id="ember14"]')
            return True
        except:
            pass
        return False 
    
    def generateUrls(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        try: 
            with open('data/urlData.txt', 'w',encoding="utf-8" ) as file:
                linkedinJobLinks = utils.LinkedinUrlGenerate().generateUrlLinks()
                for url in linkedinJobLinks:
                    file.write(url+ "\n")
            utils.prGreen("✅ Apply urls are created successfully, now the bot will visit those urls.")
        except:
            utils.prRed("❌ Couldn't generate urls, make sure you have editted config file line 25-39")

    def linkJobApply(self):
        self.generateUrls()
        countApplied = 0
        countJobs = 0

        urlData = utils.getUrlDataFile()

        for url in urlData:        
            self.driver.get(url)
            time.sleep(random.uniform(1, constants.botSpeed))

            totalJobs = self.driver.find_element(By.XPATH,'//small').text 
            totalPages = utils.jobsToPages(totalJobs)

            urlWords =  utils.urlToKeywords(url)
            lineToWrite = "\n Category: " + urlWords[0] + ", Location: " +urlWords[1] + ", Applying " +str(totalJobs)+ " jobs."
            self.displayWriteResults(lineToWrite)

            processed = set()
            for page in range(totalPages):
                currentPageJobs = constants.jobsPerPage * page
                url = url +"&start="+ str(currentPageJobs)
                self.driver.get(url)
                time.sleep(random.uniform(3, constants.botSpeed))

                offersPerPage = self.driver.find_elements(By.XPATH, '//li[@data-occludable-job-id]')
                offerIds = []
                time.sleep(random.uniform(1, constants.botSpeed))

                for offer in offersPerPage:
                    if not self.element_exists(offer, By.XPATH, ".//li[contains(., 'Applied')]"):
                        offerId = offer.get_attribute("data-occludable-job-id")
                        # offerId = offer.get_attribute("data-job-id")
                        if offerId not in processed:
                            offerIds.append(int(offerId.split(":")[-1]))
                            processed.add(offerId)

                for jobID in offerIds:
                    offerPage = 'https://www.linkedin.com/jobs/view/' + str(jobID)
                    self.driver.get(offerPage)
                    time.sleep(random.uniform(1, constants.botSpeed))

                    countJobs += 1

                    jobProperties = self.getJobProperties(countJobs)
                    jobRequirements = self.getJobRequirements()
                    if jobRequirements:
                        lineToWrite = jobProperties + " | " + f"* 🤬 Requirements not match Job({jobRequirements}), skipped!: " +str(offerPage)
                        self.displayWriteResults(lineToWrite)
                    elif "blacklisted" in jobProperties:
                        lineToWrite = jobProperties + " | " + "* 🤬 Blacklisted Job, skipped!: " +str(offerPage)
                        self.displayWriteResults(lineToWrite)
                    
                    else :                    
                        easyApplybutton = self.easyApplyButton()

                        if easyApplybutton is not False:
                            easyApplybutton.click()
                            time.sleep(random.uniform(1, constants.botSpeed))
                            
                            try:
                                self.chooseResume()
                                self.driver.find_element(By.CSS_SELECTOR, "button[aria-label='Submit application']").click()
                                time.sleep(random.uniform(1, constants.botSpeed))

                                lineToWrite = jobProperties + " | " + "* 🥳 Just Applied to this job: "  +str(offerPage)
                                self.displayWriteResults(lineToWrite)
                                countApplied += 1

                            except:
                                try:
                                    self.driver.find_element(By.CSS_SELECTOR,"button[aria-label='Continue to next step']").click()
                                    time.sleep(random.uniform(1, constants.botSpeed))
                                    self.chooseResume()
                                    # comPercentage = self.driver.find_element(By.XPATH,'html/body/div[3]/div/div/div[2]/div/div/span').text
                                    comPercentage = self.driver.find_element(By.CSS_SELECTOR, "div.display-flex span").text
                                    percenNumber = int(comPercentage[0:comPercentage.index("%")])
                                    result = self.applyProcess(percenNumber,offerPage)
                                    lineToWrite = jobProperties + " | " + result
                                    self.displayWriteResults(lineToWrite)
                                
                                except Exception as e:
                                    self.chooseResume()
                                    lineToWrite = jobProperties + " | " + "* 🥵 Cannot apply to this Job! " + str(offerPage)
                                    # lineToWrite = jobProperties + " | " + "* 🥵 Cannot apply to this Job! " +str(offerPage) + f" | {e}"
                                    self.displayWriteResults(lineToWrite)
                        else:
                            lineToWrite = jobProperties + " | " + "* 🥳 Already applied! Job: " +str(offerPage)
                            self.displayWriteResults(lineToWrite)


            utils.prYellow("Category: " + urlWords[0] + "," +urlWords[1]+ " applied: " + str(countApplied) +
                  " jobs out of " + str(countJobs) + ".")
        
        utils.donate(self)

    def chooseResume(self):
        try:
            self.driver.find_element(
                By.CLASS_NAME, "jobs-document-upload__title--is-required")
            resumes = self.driver.find_elements(
                By.XPATH, "//div[contains(@class, 'ui-attachment--pdf')]")
            if (len(resumes) == 1 and resumes[0].get_attribute("aria-label") == "Select this resume"):
                resumes[0].click()
            elif (len(resumes) > 1 and resumes[config.preferredCv-1].get_attribute("aria-label") == "Select this resume"):
                resumes[config.preferredCv-1].click()
            elif (type(len(resumes)) != int):
                utils.prRed(
                    "❌ No resume has been selected please add at least one resume to your Linkedin account.")
        except:
            pass

    def getJobProperties(self, count):
        textToWrite = ""
        jobTitle = ""
        jobLocation = ""

        try:
            jobTitle = self.driver.find_element(By.XPATH, "//h1[contains(@class, 't-24')]").get_attribute("innerHTML").strip()
            res = [blItem for blItem in config.blackListTitles if (blItem.lower() in jobTitle.lower())]
            if (len(res) > 0):
                jobTitle += "(blacklisted title: " + ' '.join(res) + ")"
        except Exception as e:
            if (config.displayWarnings):
                utils.prYellow("⚠️ Warning in getting jobTitle: " + str(e)[0:50])
            jobTitle = ""

        try:
            time.sleep(5)
            jobDetail = self.driver.find_element(By.XPATH, "//div[@class='job-details-jobs-unified-top-card__company-name']")
            jobDetail = jobDetail.find_element(By.XPATH, ".//a").text \
                if len(jobDetail.find_elements(By.XPATH, ".//a")) else jobDetail.text
            jobDetail = ''.join(jobDetail)
            if jobDetail in config.blacklistCompanies:
                jobDetail += "(blacklisted company: " + jobDetail + ")"
        except Exception as e:
            if (config.displayWarnings):
                print(e)
                utils.prYellow("⚠️ Warning in getting jobDetail: " + str(e)[0:100])
            jobDetail = ""

        try:
            jobRequirements = self.driver.find_element(By.XPATH, "//ul[contains(@class, 'job-details-about-the-job-module__requirements-list')]//li").text
            jobReqText = ''.join(jobRequirements)
            if "sponsorship" in jobReqText:
                jobDetail += "(cannot sponsor company: " + jobReqText + ")"
        except Exception as e:
            pass

        try:
            jobWorkStatusSpans = self.driver.find_elements(By.XPATH, "//span[contains(@class,'ui-label ui-label--accent-3 text-body-small')]//span[contains(@aria-hidden,'true')]")
            for span in jobWorkStatusSpans:
                jobLocation = jobLocation + " | " + span.text

        except Exception as e:
            if (config.displayWarnings):
                print(e)
                utils.prYellow("⚠️ Warning in getting jobLocation: " + str(e)[0:100])
            jobLocation = ""

        textToWrite = str(count) + " | " + jobTitle +" | " + jobDetail + jobLocation
        return textToWrite

    def getJobRequirements(self):
        resp = ""
        not_matches = []
        try:
            jobRequirements = self.driver.find_elements(By.XPATH, "//ul[contains(@class, 'job-details-about-the-job-module__requirements-list')]//li")
            if jobRequirements:
                for r in jobRequirements:
                    jobReqText = ''.join(r.text).lower()
                    not_matches.extend([k for k, v in self.answers['requirements'].items() if k.lower() in jobReqText and v == "no"])
                if not_matches:
                    resp = ', '.join(not_matches)
        except Exception as e:
            print(f"problem while getting job requirements")
        return resp


    def easyApplyButton(self):
        try:
            time.sleep(random.uniform(1, constants.botSpeed))
            button = self.driver.find_element(By.XPATH, "//div[contains(@class,'jobs-apply-button--top-card')]//button[contains(@class, 'jobs-apply-button')]")
            EasyApplyButton = button
        except: 
            EasyApplyButton = False

        return EasyApplyButton

    def applyProcess(self, percentage, offerPage):
        applyPages = math.floor(100 / percentage) - 2 
        result = ""
        for pages in range(applyPages):  
            self.driver.find_element(By.CSS_SELECTOR, "button[aria-label='Continue to next step']").click()
        # input fields
        div_questions = self.driver.find_elements(By.XPATH, "//div[@class='artdeco-text-input--container ember-view']")
        if div_questions:
            for d in div_questions:
                q = d.find_element(By.XPATH, ".//label").text
                print(f"q: {q}")
                for k, v in self.answers['inputField'].items():
                    if k in q:
                        input_ = d.find_element(By.XPATH, ".//input")
                        input_.clear()
                        input_.send_keys(v)
                        print(f"✅ 填入 {v}: {q}")
                        break
        # fieldset
        # $x("//fieldset//span[@aria-hidden='true']")
        fieldsets = self.driver.find_elements(By.XPATH, "//fieldset")
        if fieldsets:
            for f in fieldsets:
                q = f.find_element(By.XPATH, ".//span[@aria-hidden='true']").text
                print(f"q: {q}")
                for k, v in self.answers['radio'].items():
                    if k in q:
                        radios = f.find_elements(By.XPATH, ".//label")
                        for r in radios:
                            if r.get_attribute("data-test-text-selectable-option__label").lower() == v.lower():
                                r.click()
                                print(f"✅ Radio 選擇 {v}: {q}")
                                break
        # Handle dropdown/select elements
        selects = self.driver.find_elements(By.XPATH, "//select[contains(@data-test-text-entity-list-form-select,'')]")
        if selects:
            for select_elem in selects:
                try:
                    # Get the question text from the parent div's label or the select's aria-describedby
                    try:
                        q = select_elem.find_element(By.XPATH, "./ancestor::div[contains(@class, 'fb-dropdown')]//label").text
                    except:
                        # If no label found, try to get the question from aria-describedby
                        error_id = select_elem.get_attribute('aria-describedby')
                        if error_id:
                            q = error_id.replace('-error', '').split('-')[-1]  # Extract the question from the ID
                        else:
                            q = select_elem.get_attribute('id').split('-')[-1]  # Fallback to ID
                    
                    print(f"Dropdown q: {q}")
                    
                    # First check if any option matches our answers
                    select = Select(select_elem)
                    options = [option.text.strip() for option in select.options]
                    
                    # Skip the default "Select an option" if it exists
                    if "Select an option" in options:
                        options.remove("Select an option")
                    
                    # Try to find matching answer from our config
                    answer_found = False
                    for k, v in self.answers['dropdown'].items():
                        if k.lower() in q.lower():
                            # Try to find exact match first
                            if v in options:
                                select.select_by_visible_text(v)
                                print(f"✅ Dropdown exact match 選擇 {v}: {q}")
                                answer_found = True
                                break
                            # Try partial match
                            for option in options:
                                if v.lower() in option.lower():
                                    select.select_by_visible_text(option)
                                    print(f"✅ Dropdown partial match 選擇 {option}: {q}")
                                    answer_found = True
                                    break
                            if answer_found:
                                break
                    
                    # If no match found from answers, select first non-default option if available
                    if not answer_found and options:
                        select.select_by_visible_text(options[0])
                        print(f"✅ Dropdown default 選擇 {options[0]}: {q}")
                
                except Exception as e:
                    if config.displayWarnings:
                        print(f"Warning in handling dropdown: {str(e)}")
                    continue

        self.driver.find_element( By.CSS_SELECTOR, "button[aria-label='Review your application']").click()
        time.sleep(random.uniform(1, constants.botSpeed))

        if config.followCompanies is False:
            try:
                self.driver.find_element(By.CSS_SELECTOR, "label[for='follow-company-checkbox']").click()
            except:
                pass

        self.driver.find_element(By.CSS_SELECTOR, "button[aria-label='Submit application']").click()
        time.sleep(random.uniform(1, constants.botSpeed))

        result = "* 🥳 Just Applied to this job: " + str(offerPage)

        return result

    def displayWriteResults(self,lineToWrite: str):
        try:
            print(lineToWrite)
            utils.writeResults(lineToWrite)
        except Exception as e:
            utils.prRed("❌ Error in DisplayWriteResults: " +str(e))

    def element_exists(self, parent, by, selector):
        for _ in range(5):  # 最多重試5次
            try:
                return len(parent.find_elements(by, selector)) > 0
            except selenium.common.exceptions.StaleElementReferenceException:
                time.sleep(1)
                pass

start = time.time()
Linkedin().linkJobApply()
end = time.time()
utils.prYellow("---Took: " + str(round((time.time() - start)/60)) + " minute(s).")
