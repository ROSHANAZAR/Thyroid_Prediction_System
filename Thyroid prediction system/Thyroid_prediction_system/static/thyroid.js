const xhr = new XMLHttpRequest();
xhr.open('GET', 'http://127.0.0.1:5000/Accuracy');
xhr.onload = function() {

  

var dec_avg=document.getElementById("Dec_Avg_Acc");
var dec_max=document.getElementById("Dec_Max_Acc");
var ran_avg=document.getElementById("Ran_Avg_Acc");
var ran_max=document.getElementById("Ran_Max_Acc");
var log_avg=document.getElementById("Log_Avg_Acc");
var log_max=document.getElementById("Log_Max_Acc");
var naive_avg=document.getElementById("Naive_Avg_Acc");
var naive_max=document.getElementById("Naive_Max_Acc");
spinner2=document.getElementById("spinner2")


  if (xhr.status === 200) {
    const data = JSON.parse(xhr.responseText);
    console.log(data);
    console.log(data.Naive_avg);
    console.log(data.Naive_max);
    dec_avg.innerHTML=data.Dec_Avg;
    dec_max.innerHTML=data.Dec_max;
    ran_avg.innerHTML=data.Ran_Avg;
    ran_max.innerHTML=data.Ran_max;
    log_avg.innerHTML=data.Log_Avg;
    log_max.innerHTML=data.Log_max;
    naive_avg.innerHTML=data.Naive_Avg;
    naive_max.innerHTML=data.Naive_max;
    spinner2.style.visibility="hidden"
  } 
  
  else {
    console.error('Request failed. Returned status:', xhr.status);
  }
};
xhr.onerror = function() {
  console.error('Request failed. Unable to connect to server.');
};
xhr.send();






var predict=document.getElementById("predict");
var accuracy=document.getElementById("compare");
var prediction_slide=document.getElementById("prediction");
var accuracy_slide=document.getElementById("accuracy");

predict_button.addEventListener("click",function(){

var Name=document.getElementById("name").value;
var Age=document.getElementById("age").value;
var Gender=document.getElementById("gender").value;
var Pregnancy=document.getElementById("pregnancy").value;
var T3=document.getElementById("t3").value;
var T4=document.getElementById("t4").value;
var TSH=document.getElementById("tsh").value;
predict_button=document.getElementById("predict_button")
var Status=document.getElementById("status");
var dec_avg=document.getElementById("Dec_Avg_Acc");
var dec_max=document.getElementById("Dec_Max_Acc");
var ran_avg=document.getElementById("Ran_Avg_Acc");
var ran_max=document.getElementById("Ran_Max_Acc");
var log_avg=document.getElementById("Log_Avg_Acc");
var log_max=document.getElementById("Log_Max_Acc");
var naive_avg=document.getElementById("Naive_Avg_Acc");
var naive_max=document.getElementById("Naive_Max_Acc");
var spinner=document.getElementById("spinner")
spinner.style.visibility="inherit";

  console.log(Name,Age,Pregnancy,Gender,T3,T4,TSH,Status)


  const data = { name: Name,age: Age, gender: Gender,pregnancy: Pregnancy,t3: T3,t4: T4,tsh: TSH };
  const xhr = new XMLHttpRequest();
  xhr.open('POST', 'http://127.0.0.1:5000/Predict');
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onload = function() {
    if (xhr.status === 200) {
       const response = JSON.parse(xhr.responseText);
       console.log(response);
        dec_avg.innerHTML=response.Dec_Avg;
    dec_max.innerHTML=response.Dec_max;
    ran_avg.innerHTML=response.Ran_Avg;
    ran_max.innerHTML=response.Ran_max;
    log_avg.innerHTML=response.Log_Avg;
    log_max.innerHTML=response.Log_max;
    naive_avg.innerHTML=response.Naive_Avg;
    naive_max.innerHTML=response.Naive_max;
       
       Status.innerHTML=response.status

   spinner.style.visibility="hidden";

  } else {
       console.error('Request failed. Returned status:', xhr.status);
  }
  };
    xhr.onerror = function() {
      console.error('Request failed. Unable to connect to server.');
  };
   xhr.send(JSON.stringify(data));


});






predict.addEventListener("click",function(){
    predict.style.textDecoration="underline"
    accuracy.style.textDecoration="none"
    prediction_slide.style.visibility="inherit"
    accuracy_slide.style.visibility="hidden"
});
accuracy.addEventListener("click",function(){
predict.style.textDecoration="none"
    accuracy.style.textDecoration="underline"

    prediction_slide.style.visibility="hidden"
    accuracy_slide.style.visibility="inherit"
});
