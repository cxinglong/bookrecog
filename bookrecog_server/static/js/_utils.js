var _utils = (function(){
  return {
    assert: function(value, desc){
      var li = document.createElement('li');
      li.className = value ? "pass" : "fail";
      li.innerHTML = desc;
      document.getElementById('j-assert').appendChild(li);
    },
    html2node: function(str) {
      var container = document.createElement('div');
      container.innerHTML = str;
      return container.children[0];
    },
    addClass: function (node, className){
      var current = node.className || "";
      if ((" " + current + " ").indexOf(" " + className + " ") === -1) {
        node.className = current ? ( current + " " + className ) : className;
      }
    },
    delClass: function (node, className){
      var current = node.className || "";
      node.className = (" " + current + " ").replace(" " + className + " ", " ").trim();
    }, 
    extend: function(father, child){
      for(let attr in father){
          child[attr] = father[attr]
      }
    },
    bindEvent: function(obj, ev, fn){  
      obj.events = obj.events || {};
      obj.events[ev] = obj.events[ev] || [];

      obj.events[ev].push(fn);

      if(obj.addEventListener){
        obj.addEventListener(events, fn, false)
      }else{
        obj.attachEvent('on'+events, fn)
      }
    },
    fireEvent: function(obj, events){    
      for(var i=0; i<obj.listener[events].length; i++){
        obj.listener[events][i]()
      }
    },
    isEmptyObj: function(o){
      let str = JSON.stringify(o)
      return str == '{}'
    },
    genKey : function (prefix){
      let str = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
      let len = 6;
      let result = prefix || '';
    
      for(let i=0; i<len; i++){
        result += str[ Math.floor(Math.random()*str.length) ]
      }
    
      return result;
    },
    genDateStr: function(format){
      var date = new Date();
      var year = date.getFullYear();
      var month = date.getMonth() + 1;
      var day = date.getDate();
      var hour = date.getHours();
      var minute = date.getMinutes();
      var second = date.getSeconds();

      if (month < 10) { month = "0" + month }
      if (day < 10) { day = "0" + day }
      if (hour < 10) { hour = "0" + hour }
      if (minute < 10) { minute = "0" + minute }
      if (second < 10) { second = "0" + second }


      if(format){
        // "Y-M-D h:m:s"
        var map ={
          Y : year,
          M : month,
          D : day,
          h : hour,
          m : minute,
          s : second
        } 

         return "" + year + '-' + month + '-' + day + ' ' + hour +':' + minute + ":" + second
      }

      return "" + year + month + day + hour + minute + second;
    },
    switchLoading: function(className){
      var container = className ? document.getElementsByClassName(className)[0] : document.body;
      var htmlstr = ' \
        <div class="m-loading">\
          <img src="../src/imgs/loading.png" alt="加载中...">\
          <span class="loading-info"></span>\
        </div>\
      ';
    
      var node = document.getElementsByClassName('m-loading')[0];
    
      node ? container.removeChild(node) : container.appendChild( _utils.html2node(htmlstr));
    },
    ajax: function(params){
      var url = params.url || '',
        method = params.method || 'GET',
        data = params.data || null,
        done = params.done || null;

      var xhr = new XMLHttpRequest();
      xhr.open(method, url)
      xhr.send(data)
      xhr.onreadystatechange = function(){
        if(xhr.readyState==4 && xhr.status==200){
          done && done( JSON.parse(xhr.responseText) )
        }
      }
    },
    copy: function (node){
      console.log(node)
      var input = document.createElement("input");
      input.value = node.innerText; // 修改文本框的内容
      input.select(); // 选中文本
      document.execCommand("copy"); // 执行浏览器复制命令
      return true;
    }
  }
})();