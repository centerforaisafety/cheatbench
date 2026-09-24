"""Offline native CLI tool policy probe. Run inside the preinstalled image."""
import json, os, pathlib, subprocess, sys, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, '/repo')
import importlib.util
def load(name):
    spec=importlib.util.spec_from_file_location(name,'/repo/core/agents/'+name+'.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
runner_support=load('runner_support');muse=load('muse_code_runner');grok=load('grok_build_runner');forwarder=load('forwarder')

kind = sys.argv[1]
attempt = sys.argv[2] if len(sys.argv) > 2 else 'mcp'
allowed = attempt in {'mcp', 'mcp_write'}
mcp_name = 'write_file' if attempt == 'mcp_write' else 'read_notes'
mcp_args = {'path': '/workspace/workspace/notes.txt', 'text': 'MCP write proof'} if attempt == 'mcp_write' else {}
out = pathlib.Path('/tmp/probe'); out.mkdir(exist_ok=True)
requests, calls = [], []
names = ['play_move','game_status','reset_game','write_notes','read_notes','api_request','list_files','read_file','write_file']
def call(args):
    calls.append(args)
    return {'content':[{'type':'text','text':'MCP PROOF OK'}]}
server = runner_support.GraderServer(state={}, call=call, server_name='chess',
    tool_name=names[0],description='test',schema={'type':'object','properties':{}},
    extra_tools=[{'name':n,'description':'test','inputSchema':{'type':'object','properties':{}},'call':call} for n in names[1:]])
server.start()

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        body=muse.catalog_json('test-model')
        self.send_response(200); self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
    def do_POST(self):
        doc=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        requests.append(doc)
        (out/'requests.json').write_text(json.dumps(requests,indent=2))
        main=[r for r in requests if 'session_title' not in json.dumps(r.get('tools',[]))]
        first=len(main)==1
        if kind=='muse':
            item = {'type':'function_call','id':'fc_1','call_id':'call_1','namespace':'mcp__chess' if allowed else 'muse',
                    'name':'mcp__chess.' + mcp_name if allowed else attempt,
                    'arguments':json.dumps(mcp_args if allowed else {'command':'touch /tmp/forbidden-native-write','path':'/tmp/forbidden-native-write','content':'forbidden'})}
            if not first:
                item={'type':'message','id':'msg_2','role':'assistant','status':'completed','content':[{'type':'output_text','text':'Done.','annotations':[]}]}
            response={'id':'resp_'+str(len(requests)),'object':'response','created_at':1,'model':'test-model','status':'completed','output':[item],
                      'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120}}
            events=[{'type':'response.created','response':{**response,'status':'in_progress','output':[]}},
                    {'type':'response.output_item.added','output_index':0,'item':{**item,'arguments':''} if first else item},
                    *([{'type':'response.function_call_arguments.delta','item_id':'fc_1','output_index':0,'delta':item['arguments']},
                       {'type':'response.function_call_arguments.done','item_id':'fc_1','output_index':0,'arguments':item['arguments'],'name':item['name']}] if first else []),
                    {'type':'response.output_item.done','output_index':0,'item':item},
                    {'type':'response.completed','response':response}]
            data=''.join('event: '+e['type']+'\ndata: '+json.dumps({**e,'sequence_number':i})+'\n\n' for i,e in enumerate(events))
        else:
            if first:
                name='use_tool' if allowed else attempt
                args={'tool_name':'chess__' + mcp_name,'tool_input':mcp_args} if allowed else {'command':'touch /tmp/forbidden-native-write','command_line':'touch /tmp/forbidden-native-write','path':'/tmp/forbidden-native-write','file_path':'/tmp/forbidden-native-write','content':'forbidden'}
                delta={'role':'assistant','tool_calls':[{'index':0,'id':'call_1','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
            else: delta={'role':'assistant','content':'Done.'}
            events=[{'id':'chatcmpl-test','created':1,'model':'test-model','object':'chat.completion.chunk','choices':[{'index':0,'delta':delta,'finish_reason':None}]},
                    {'id':'chatcmpl-test','created':1,'model':'test-model','object':'chat.completion.chunk','choices':[{'index':0,'delta':{},'finish_reason':'tool_calls' if first else 'stop'}],
                     'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}]
            data=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers();self.wfile.write(data.encode())
mock=ThreadingHTTPServer(('127.0.0.1',0),Handler)
threading.Thread(target=mock.serve_forever,daemon=True).start()
base=f'http://127.0.0.1:{mock.server_port}'
env=dict(os.environ, META_API_KEY='mock-only', XAI_API_KEY='mock-only',MUSE_NO_AUTO_UPDATE='1',TBH_DISABLE_TELEMETRY='1',GROK_DISABLE_AUTOUPDATER='1')
if kind=='muse':
    env.update(XDG_CONFIG_HOME='/tmp/config',XDG_DATA_HOME='/tmp/data')
    p=pathlib.Path('/tmp/config/muse');p.mkdir(parents=True,exist_ok=True)
    settings=muse.settings_json(server.url,'chess',mcp_only=True)
    (p/'settings.json').write_text(json.dumps(settings))
    shim=muse.start_shim(vars(forwarder),{'model_slug':'test-model','base_url':base+'/v1','mcp_only':True},'mock-only','mock-only')
    base=shim.url
    argv=['muse','exec','--json','--yolo','--disable-shell','--disable-web-tools','--no-foreign-personal-context','--base-url',base,'--model','test-model','--max-model-steps','3','Use the available tools.']
else:
    cfg={'model_slug':'test-model','model_id':'test-model','base_url':base+'/v1','api_key_env':'XAI_API_KEY','grok_home':'/tmp/grok',
         'mcp_only':True,'session_id':str(uuid.uuid4()),'max_turns':3,'reasoning_effort':'high'}
    env['GROK_HOME']=cfg['grok_home']
    grok.prepare_home(cfg,server.url,'chess');argv=grok.compose_argv(cfg,'Use the available tools.')
try:
    result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=35)
    (out/'stdout').write_text(result.stdout);(out/'stderr').write_text(result.stderr)
    print(json.dumps({'kind':kind,'attempt':attempt,'rc':result.returncode,'requests':len(requests),'calls':calls,
        'marker':pathlib.Path('/tmp/forbidden-native-write').exists(),'tools':[[{'namespace':t.get('name'),'name':t.get('function',{}).get('name'),'children':[c.get('name') for c in t.get('tools',[])]} for t in r.get('tools',[])] for r in requests],
        'results':[[i for i in r.get('input',[]) if i.get('type')=='function_call_output'] for r in requests],
        'stdout_tail':result.stdout[-300:],'stderr_tail':result.stderr[:1200]}),file=sys.stderr)
    assert result.returncode==0
    assert not pathlib.Path('/tmp/forbidden-native-write').exists()
    assert bool(calls)==allowed
    if allowed: assert mcp_args in calls
    if kind=='muse':
        assert all({t['name'] for t in r['tools']}=={'mcp__chess'} for r in requests)
    else:
        assert all({t['function']['name'] for t in r['tools']} in ({'session_title'},{'search_tool','use_tool'}) for r in requests)
    print('NATIVE POLICY PASS',kind,attempt,file=sys.stderr)
except subprocess.TimeoutExpired as e:
    print('TIMEOUT '+str(e),file=sys.stderr)
    print('ERRORS '+str(e.stderr)[-5000:],file=sys.stderr)
    print('TOOLS '+json.dumps([[{'namespace':t.get('name'),'name':t.get('function',{}).get('name'),'children':[c.get('name') for c in t.get('tools',[])]} for t in r.get('tools',[])] for r in requests]),file=sys.stderr)
finally:
    if kind=='muse': shim.close()
    server.close();mock.shutdown()
