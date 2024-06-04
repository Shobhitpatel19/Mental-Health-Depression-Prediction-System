import "./style.css";
import messages from "./messages.json";

function ContactUsSection() {
    return (
        <div>
            <div className="contact-section">
                <div>
                     <p className="standout-heading1">{messages["cont"]}</p>
                </div>
                <div>
                     <p className="standout-heading2">{messages["act_us"]}</p>
                </div>
            </div>

            <div className="contact-info-section">
                <p className="contact-header-text-style">{messages["need_get_touch"]}</p>
                <p className="contact-header-text-style1">{messages["we_here_you"]}</p>
            </div>

        </div>
    );

}

export default ContactUsSection;