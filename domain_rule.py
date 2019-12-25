import os
import re
import json

dic_path = 'dataSet/op_dic/dic'
op_path = 'dataSet/op_dic'


def app(text, domain, all_app, app_op):

    # print(all_app)
    for app in all_app:
        new_text = ['','']
        if text == app:
            return ['app',"LAUNCH"]
            # result[i]['intent'] = "LAUNCH"
        if domain != 'app':
            for op in app_op:
                op_search = re.search(op.strip(),text)
                if op_search:
                    new_text = [text[:op_search.span()[0]],text[op_search.span()[1]:]]
                    break
            search_words = re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",app.strip())
            if op_search and (re.search(search_words,new_text[0]) or re.search(search_words,new_text[-1])):
                return ['app']
                # result[i]['domain'] = 'app'
    return []


def music(text, domain, music_op):

    if domain != 'music':
        for op in music_op:
            op_search = re.search(op.strip(),text)
            if op_search:
                text = text[:op_search.span()[0]] + text[op_search.span()[1]:]
                return ['music']
    return []


def poetry(text, domain, all_poetry, poetry_op):

    for poetry in all_poetry:
        new_text = ['','']
        if text == poetry:
            return ['poetry',"QUERY"]
            # result[i]['intent'] = "LAUNCH"
        if domain != 'poetry':
            for op in poetry_op:
                op_search = re.search(op.strip(),text)
                if op_search:
                    new_text = [text[:op_search.span()[0]], text[op_search.span()[1]:]]
                    break
            search_words = re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",poetry.strip())
            if op_search and (re.search(search_words,new_text[0]) or re.search(search_words,new_text[-1])):
                return ['poetry']
    return []


def bus(text, domain, all_city):
    for city in all_city:
        new_text = ['','']
        if domain != 'bus':
            for tool in ['车汽','车']:
                text = text[::-1]
                tool_search = re.search(tool,text)
                if tool_search:
                    text = text[:tool_search.span()[0]] + text[tool_search.span()[1]:]
                    text = text[::-1]
                    break
            for op in ['到','去','回']:
                op_search = re.search(op, text)
                if op_search:
                    new_text = [text[:op_search.span()[0]], text[op_search.span()[1]:]]
                    break
            if tool_search and op_search and (re.search(city.strip(), new_text[0]) or re.search(city.strip(), new_text[1])):
                    return ['bus']
    return []


def video(text, domain, videos):
    subtexts = []
    for k in range(len(text)):
        for i in range(k,len(text)):
            subtexts.append(text[k:i+1])

    if videos.get(text,None):
        return ['video', text]
    for subtext in subtexts:
        if domain != 'video':
            search_result = videos.get(subtext,None)
            if search_result:
                th = float(len(subtext))/float(len(text))
                if th >= 0.6:
                    return 'video'
    return []



def tvchannel(text, domain, all_tvchannel):
    for tvchannel in all_tvchannel:
        final_result = ['tvchannel',None,None]
        if text == tvchannel:
            return ['tvchannel', text]
        if domain != 'tvchannel':
            search_words = re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", tvchannel.strip())
            search_result = re.search(search_words,text)
            if search_result:
                resolution_search = re.search('高清',text)
                if resolution_search:
                    text = text[:resolution_search.span()[0]]+text[resolution_search.span()[1]:]
                    final_result = ['tvchannel','高清',search_words]
                th = float(search_result.span()[1] - search_result.span()[0])/float(len(text))
                if th >= 0.7:
                    return final_result
    return []


def website(text,domain,websites,website_op):
    subtexts = []
    for k in range(len(text)):
        for i in range(k,len(text)):
            subtexts.append(text[k:i+1])

    if websites.get(text,None):
        return ['website', text]

    for subtext in subtexts:
        new_text = ['','']
        if domain != 'website':
            for op in website_op:
                op_search = subtext.find(op)
                if op_search != -1:
                    s = subtext.find(op)
                    new_text = [subtext[:s],subtext[s+len(op)-1:]]
                    break
            if op_search and (websites.get(new_text[0]) or websites.get(new_text[-1])):
                return ['website',subtext]
                # result[i]['domain'] = 'app'
    return []



##################################################################################################################

def domain_rule(result):

    app_count = 0
    for i,pred in enumerate(result):
        flag = False
        text = pred['text']
        domain = pred['domain']

        # app up
        with open(os.path.join(dic_path,'app.txt'),encoding='UTF-8') as f_app:
            all_app = f_app.read().strip().split('\n')

        with open(os.path.join(op_path,'app_op.txt'),encoding='UTF-8') as f_app_op:
            app_op = f_app_op.read().strip().split('\n')

        app_result = app(text, domain, all_app, app_op)
        if len(app_result) == 2:
            result[i]['domain'] = app_result[0]
            result[i]['intent'] = app_result[1]
            continue
        elif len(app_result) == 1:
            result[i]['domain'] = app_result[0]
            continue
        else:
            pass



        # music down
        # with open(os.path.join(op_path,'music_op.txt'),encoding='UTF-8') as f_music_op:
        #     music_op = f_music_op.read().strip().split('\n')
        #
        # music_result = music(text,domain,music_op)
        # if len(music_result) == 1:
        #     result[i]['domain'] = music_result[0]
        #     continue
        # else:
        #     pass


        # poetry no change
        # with open(os.path.join(dic_path,'poetry.txt'),encoding='UTF-8') as f_poetry:
        #     all_poetry = f_poetry.read().strip().split('\n')
        #
        # with open(os.path.join(op_path,'poetry_op.txt'),encoding='UTF-8') as f_poetry_op:
        #     poetry_op = f_poetry_op.read().strip().split('\n')
        #
        # poetry_result = poetry(text, domain,all_poetry, poetry_op)
        # if len(poetry_result) == 2:
        #     result[i]['domain'] = poetry_result[0]
        #     result[i]['intent'] = poetry_result[1]
        #     continue
        # elif len(poetry_result) == 1:
        #     result[i]['domain'] = poetry_result[0]
        #     continue
        # else:
        #     pass


        # bus no change
        with open(os.path.join(dic_path,'city.txt'),encoding='UTF-8') as f_city:
            all_city = f_city.read().strip().split('\n')

        city_result = bus(text,domain,all_city)
        if len(city_result) == 1:
            result[i]['domain'] = city_result[0]
            continue
        else:
            pass


        # video up
        with open(os.path.join(dic_path,'film.txt'),encoding='UTF-8') as f_video:
            all_video = f_video.read().strip().split('\n')
            videos = {}
            for v in all_video:
                videos[re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",v.strip())] = 1

        with open(os.path.join(dic_path,'film_category.txt'),encoding='UTF-8') as f_video:
            video_category = f_video.read().strip().split('\n')

        video_result = video(text,domain,videos)
        if len(video_result) == 2:
            result[i]['domain'] = video_result[0]
            if video_result[1] in video_category:
                result[i]['slots']['category'] = video_result[1]
            else:
                result[i]['slots']['name'] = video_result[1]
            continue
        if len(video_result) == 1:
            result[i]['domain'] = video_result[0]
            continue
        else:
            pass


        # tvchannel up
        with open(os.path.join(dic_path,'tvchannel.txt'),encoding='UTF-8') as f_tvchannel:
            all_tvchannel = f_tvchannel.read().strip().split('\n')

        tvchannel_result = tvchannel(text,domain,all_tvchannel)
        if len(tvchannel_result) == 3:
            result[i]['domain'] = tvchannel_result[0]
            result[i]['slots']['resolution'] = tvchannel_result[1]
            result[i]['slots']['name'] = tvchannel_result[2]
            continue
        if len(tvchannel_result) == 2:
            result[i]['domain'] = tvchannel_result[0]
            result[i]['slots']['name'] = tvchannel_result[1]
            continue
        else:
            pass


        # website down
        with open(os.path.join(dic_path,'website.txt'),encoding='UTF-8') as f_website:
            all_website = f_website.read().strip().split('\n')
            websites = {}
            for w in websites:
                websites[re.sub("[\s+\.\!\/_,$%^*()+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",w.strip())] = 1

        with open(os.path.join(op_path,'website_op.txt'),encoding='UTF-8') as f_website_op:
            website_op = f_website_op.read().strip().split('\n')

        website_result = website(text, domain, websites, website_op)
        if len(website_result) == 2:
            result[i]['domain'] = website_result[0]
            result[i]['slots']['name'] = website_result[1]
            continue
        # elif len(website_result) == 1:
        #     result[i]['domain'] = website_result[0]
        #     continue
        else:
            pass





    return result


# if __name__ == '__main__':
#     result = json.load(open('result/test_result2.json', encoding = 'utf8'), object_pairs_hook = OrderedDict)
#     domain_rule(result)
